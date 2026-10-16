#!/usr/bin/env python3
"""Read-only: compare official Task1 ACT rollout with all 500 demonstration starts.

Input: a telemetry JSONL copied from the scored ROS bridge.
Data: official LeRobot v3 parquet files, no video decoding or CUDA needed.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path("/workspace/openlet/datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick")
OUT=Path("/workspace/openlet/code/LeTools-Learning/outputs/eval/task1_act_online_initial_domain_gap.json")

def mae(a,b):
    return float(np.mean(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))

def ensure(cond,msg):
    if not cond:raise RuntimeError(msg)

def take_vector(column,indices):
    return np.asarray(column.take(pa.array(indices)).to_pylist(),dtype=np.float64)

def load_demo():
    files=sorted((ROOT/"data").rglob("*.parquet"))
    ensure(files,"No LeRobot v3 parquet files under "+str(ROOT/"data"))
    starts={};first_actions={};all_states=[];all_episode_ids=[];meta={}
    rows=0
    for path in files:
        table=pq.read_table(path,columns=["episode_index","observation.state","action"])
        episode=np.asarray(table["episode_index"].to_numpy(),dtype=np.int64)
        if not len(episode):continue
        index=np.where(np.r_[True,episode[1:]!=episode[:-1]])[0]
        samples=take_vector(table["observation.state"],index)
        act=take_vector(table["action"],index)
        for i,v,a in zip(index,samples,act):
            ep=int(episode[i])
            if ep not in starts:
                starts[ep]=v;first_actions[ep]=a
        every=np.arange(0,len(episode),10)
        all_states.append(take_vector(table["observation.state"],every))
        all_episode_ids.append(episode[every])
        rows+=len(episode)
        print("[DATA] parquet={} rows={} starts={} total_rows={}".format(
            path.name,len(episode),len(starts),rows),flush=True)
    ensure(all(len(v)==18 for v in starts.values()),"dataset state is not 18D")
    ensure(all(len(v)==21 for v in first_actions.values()),"dataset action is not 21D")
    ensure(len(starts)==500,"Expected 500 demonstration episodes; found "+str(len(starts)))
    return starts,first_actions,np.concatenate(all_states),np.concatenate(all_episode_ids),files,rows

def load_rollout(path):
    ensure(path.is_file(),"rollout JSONL missing: "+str(path))
    rows=[]
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        x=json.loads(line)
        for key in ("right_arm_measured_rad","right_sg100_measured_normalized",
                    "right_arm_target_rad","base_velocity_sent"):
            ensure(key in x,"telemetry missing {} at row {}".format(key,number))
        rows.append(x)
    ensure(len(rows)>=30,"need >=30 recorded rollout frames")
    return rows

def describe_initial(starts,states,episode_ids,live):
    train_keys=sorted(k for k in starts if k<450)
    val_keys=sorted(k for k in starts if k>=450)
    ensure(len(train_keys)==450 and len(val_keys)==50,"unexpected episode coverage")
    train=np.stack([starts[k] for k in train_keys])
    val=np.stack([starts[k] for k in val_keys])
    diff_arm=np.mean(np.abs(train[:,:7]-live[None,:7]),axis=1)
    diff_hand=np.mean(np.abs(train[:,7:]-live[None,7:]),axis=1)
    diff_all=np.mean(np.abs(train-live[None,:]),axis=1)
    nearest=int(np.argmin(diff_all))
    nearest_episode=train_keys[nearest]
    trainstate=states[episode_ids<450]
    sampled_diff=np.mean(np.abs(trainstate-live[None,:]),axis=1)
    sample_closest=int(np.argmin(sampled_diff))
    closest_frame_state=trainstate[sample_closest]
    return {
        "live_first_18d_state":[round(float(x),5) for x in live],
        "train_first_frame_arm_mae_rad_min":round(float(diff_arm.min()),6),
        "train_first_frame_arm_mae_rad_median":round(float(np.median(diff_arm)),6),
        "train_first_frame_hand_mae_normalized_min":round(float(diff_hand.min()),6),
        "train_first_frame_hand_mae_normalized_median":round(float(np.median(diff_hand)),6),
        "train_first_frame_all18_mae_min":round(float(diff_all.min()),6),
        "train_first_frame_all18_mae_median":round(float(np.median(diff_all)),6),
        "closest_training_episode_start":nearest_episode,
        "closest_training_start_18d_state":[round(float(x),5) for x in train[nearest]],
        "closest_training_start_arm_mae_rad":round(float(diff_arm[nearest]),6),
        "closest_training_start_hand_mae_normalized":round(float(diff_hand[nearest]),6),
        "closest_any_training_frame_18d_mae":round(float(sampled_diff[sample_closest]),6),
        "closest_any_training_frame_18d_state":[round(float(x),5) for x in closest_frame_state],
        "heldout_first_frame_all18_mae_min":round(float(np.mean(np.abs(val-live[None,:]),axis=1).min()),6)
    }

def summarize_rollout(rows):
    first=rows[0]
    arm=np.asarray([x["right_arm_target_rad"] for x in rows],dtype=float)
    q=np.asarray([x["right_arm_measured_rad"] for x in rows],dtype=float)
    base=np.asarray([x["base_velocity_sent"] for x in rows],dtype=float)
    times=np.asarray([x["elapsed_s"] for x in rows],dtype=float)
    blocks={}
    for name,lo,hi in (("first_2s",0,2),("2_to_20s",2,20),
                        ("20_to_60s",20,60),("60_to_end",60,float("inf"))):
        mask=(times>=lo)&(times<hi)
        if not np.any(mask):continue
        v=base[mask]
        blocks[name]={
            "frames":int(np.sum(mask)),
            "base_mean_speed":round(float(np.linalg.norm(v,axis=1).mean()),8),
            "base_moving_over_0p03":int(np.sum(np.linalg.norm(v,axis=1)>0.03)),
            "arm_target_delta_mean_rad":round(float(np.mean(
                np.abs(np.diff(arm[mask],axis=0)))),6) if np.sum(mask)>1 else None,
            "arm_target_mean_tracking_error_rad":round(float(np.mean(np.abs(arm[mask]-q[mask]))),6)
        }
    return {
        "total_frames":len(rows),"duration_seconds":round(float(times[-1]),3),
        "base_total_max_speed":round(float(np.max(np.linalg.norm(base,axis=1))),8),
        "right_arm_target_total_delta_mean_rad":round(float(np.mean(np.abs(np.diff(arm,axis=0)))),7),
        "right_arm_actual_start_end_mean_change_rad":round(float(mae(q[0],q[-1])),6),
        "right_arm_target_start_end_mean_change_rad":round(float(mae(arm[0],arm[-1])),6),
        "windows":blocks,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rollout",required=True,type=Path)
    args=p.parse_args()
    rows=load_rollout(args.rollout)
    first=rows[0]
    live=np.asarray(first["right_arm_measured_rad"]+
                    first["right_sg100_measured_normalized"],dtype=float)
    ensure(live.shape==(18,),"live initial state must be 18D")
    starts,actions,states,ids,files,nrows=load_demo()
    report={
      "not_official_score":True,
      "dataset":str(ROOT),
      "data_parquet_files":len(files),
      "data_rows":nrows,
      "demonstration_episodes":len(starts),
      "recorded_online_frames":len(rows),
      "initial_state_comparison":describe_initial(starts,states,ids,live),
      "rollout_action_profile":summarize_rollout(rows),
      "data_readme_present":(ROOT/"README.md").is_file(),
      "task_metadata_present":(ROOT/"meta/tasks.jsonl").is_file()
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("\n===== ONLINE-DATASET DIAGNOSIS =====",flush=True)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("[PASS] saved:",OUT,flush=True)
    print("[NOTE] Large state distance supports mismatch; small state distance does not rule out camera/scene mismatch.",flush=True)

if __name__=="__main__":
    main()
