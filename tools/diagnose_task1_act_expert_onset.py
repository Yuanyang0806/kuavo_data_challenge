#!/usr/bin/env python3
"""Read-only expert action onset vs online ACT first action, Task1 V3.
Uses 500 local parquet demos and captured scored bridge JSONL. No ROS/GPU.
"""
import json
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path("/workspace/openlet/datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick")
TRACE=Path("/tmp/openlet_act21_rollout_194953.jsonl")
OUTPUT=Path("/workspace/openlet/code/LeTools-Learning/outputs/eval/task1_act_expert_onset_vs_online.json")
FPS=10

def metric(x):
    if not len(x):return None
    return round(float(np.median(x)),6)

def summary(v):
    if not len(v):return None
    v=np.asarray(v,dtype=np.float64)
    return {"min":round(float(v.min()),6),
            "median":round(float(np.median(v)),6),
            "p90":round(float(np.quantile(v,.90)),6),
            "max":round(float(v.max()),6)}

def group_means(a):
    x=np.asarray(a,dtype=np.float64)
    return [round(float(x[:7].mean()),5),
            round(float(x[7:18].mean()),5),
            round(float(x[18:21].mean()),5)]

def main():
    files=sorted((ROOT/"data").rglob("*.parquet"))
    if not files:raise RuntimeError("dataset parquet missing")
    if not TRACE.is_file():raise RuntimeError("expected saved online rollout "+str(TRACE))
    live=[json.loads(x) for x in TRACE.read_text().splitlines() if x.strip()]
    if len(live)<100:raise RuntimeError("saved online trajectory truncated")
    initial=live[0]
    live_state=np.array(initial["right_arm_measured_rad"]+initial["right_sg100_measured_normalized"],dtype=np.float64)
    live_first=np.array(initial["right_arm_target_rad"]+initial["right_sg100_target_normalized"]+initial["base_velocity_raw"],dtype=np.float64)
    online_v=np.array([p["base_velocity_raw"] for p in live],dtype=float)
    online_arm=np.array([p["right_arm_target_rad"] for p in live],dtype=float)
    episodes={}
    for path in files:
        t=pq.read_table(path,columns=["episode_index","observation.state","action"])
        ids=t["episode_index"].to_numpy().astype(np.int64)
        actions=np.asarray(t["action"].to_pylist(),dtype=np.float32)
        states=np.asarray(t["observation.state"].to_pylist(),dtype=np.float32)
        if actions.shape!=(len(ids),21) or states.shape!=(len(ids),18):
            raise RuntimeError("incorrect 21D/18D parquet features")
        breaks=np.flatnonzero(np.r_[True,ids[1:]!=ids[:-1],True])
        for start,end in zip(breaks[:-1],breaks[1:]):
            ep=int(ids[start])
            if ep in episodes:raise RuntimeError("episode repeated between files")
            episodes[ep]={"actions":actions[start:end],"state0":states[start].astype(float)}
    if sorted(episodes)!=list(range(500)):
        raise RuntimeError("expected exactly episodes 0..499")

    tr=list(range(450))
    starts=np.stack([episodes[e]["state0"] for e in tr])
    action0=np.stack([episodes[e]["actions"][0].astype(float) for e in tr])
    matches=np.mean(np.abs(starts-live_state[None,:]),axis=1)
    near=np.argsort(matches)[:10]
    first_action_diffs=np.mean(np.abs(action0-live_first[None,:]),axis=1)
    onsets={}
    expert_vel_all=[]
    horizons=[1,5,10,20,30,50,100]
    earliest=0
    for e in tr:
        act=episodes[e]["actions"]
        norms=np.linalg.norm(act[:,18:21],axis=1)
        first=np.flatnonzero(norms>0.03)
        onsets[e]=(round(float(first[0])/FPS,1) if len(first) else None)
        expert_vel_all.append(norms)
    firstmove=[x for x in onsets.values() if x is not None]
    expert_first={
      "arm_mean_abs_start_action_minus_start_state":summary(
        [float(np.mean(np.abs(episodes[e]["actions"][0,:7]-episodes[e]["state0"][:7]))) for e in tr]),
      "sg100_mean_abs_start_action_minus_start_state":summary(
        [float(np.mean(np.abs(episodes[e]["actions"][0,7:18]-episodes[e]["state0"][7:18]))) for e in tr]),
      "start_action_21d_mae_vs_live_first_pred":summary(first_action_diffs),
      "expert_first_base_speed_l2":summary([np.linalg.norm(episodes[e]["actions"][0,18:21]) for e in tr]),
      "base_motion_first_onset_seconds":summary(firstmove),
      "episodes_with_base_motion_over_0p03":len(firstmove),
    }
    windows={}
    for secs in horizons:
        n=secs*FPS
        v=[];arm_delta=[];hand_delta=[]
        for e in tr:
            data=episodes[e]["actions"][:n]
            if not len(data):continue
            v.extend(np.linalg.norm(data[:,18:21],axis=1).tolist())
            if len(data)>1:
                arm_delta.append(float(np.mean(np.abs(data[1:,:7]-data[:-1,:7]))))
                hand_delta.append(float(np.mean(np.abs(data[1:,7:18]-data[:-1,7:18]))))
        windows[str(secs)+"s"]={
          "expert_base_speed_mean":round(float(np.mean(v)),7),
          "expert_base_motion_fraction_over_0p03":round(float(np.mean(np.asarray(v)>0.03)),5),
          "expert_arm_action_step_delta_rad_median":metric(arm_delta),
          "expert_hand_action_step_delta_median":metric(hand_delta)}
    closest=[]
    for j in near[:5]:
        ep=tr[int(j)]
        action=action0[int(j)]
        closest.append({
          "episode":ep,
          "start_state_mae":round(float(matches[j]),7),
          "expert_first_arm_vs_live_first_pred_mae":round(float(np.mean(np.abs(action[:7]-live_first[:7]))),6),
          "expert_first_sg100_vs_live_first_pred_mae":round(float(np.mean(np.abs(action[7:18]-live_first[7:18]))),6),
          "expert_first_base_l2":round(float(np.linalg.norm(action[18:21])),6),
          "expert_first_action_base":[round(float(x),5) for x in action[18:21]],
          "expert_first_base_motion_onset_s":onsets[ep]
        })
    motion=np.linalg.norm(online_v,axis=1)
    report={
       "kind":"read_only_expert_vs_online_analysis_not_score",
       "training_episodes":450,
       "online_episode_frames":len(live),
       "online_first_action_base":[round(float(x),6) for x in live_first[18:21]],
       "online_first_arm_action_minus_state_mean_abs":round(float(np.mean(np.abs(live_first[:7]-live_state[:7]))),6),
       "online_first_hand_action_minus_state_mean_abs":round(float(np.mean(np.abs(live_first[7:18]-live_state[7:18]))),6),
       "online_first_5s_base_motion_fraction_over_0p03":round(float(np.mean(motion[:50]>.03)),6),
       "online_full_base_motion_fraction_over_0p03":round(float(np.mean(motion>.03)),6),
       "nearest_demonstration_starts":closest,
       "expert_first_action_statistics":expert_first,
       "expert_action_time_windows":windows,
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("===== EXPERT / ONLINE FIRST-ACTION COMPARISON =====",flush=True)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print("[PASS] saved",OUTPUT,flush=True)
    print("[LIMITATION] This compares joint states and action labels only. Image appearance remains unverified.",flush=True)

if __name__=="__main__":
    main()
