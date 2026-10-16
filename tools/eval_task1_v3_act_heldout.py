#!/usr/bin/env python3
"""Offline held-out action MAE: 50 unseen Task1 episodes, same frames for old/new ACT.
No ROS, no sim, no training, no checkpoint modifications.
"""
from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

LET=Path("/workspace/openlet/code/LeTools-Learning")
DATA=Path("/workspace/openlet/datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick")
OLD=LET/"outputs/train/openlet_task1_act_competition_20261008_123402/checkpoints/020000/pretrained_model"
NEW=LET/"outputs/train/openlet_task1_v3_act_finetune_20261016_144053_model/checkpoints/010000/pretrained_model"
OUT=LET/"outputs/eval/task1_v3_act_heldout_20261016.json"
EPISODES=list(range(450,500))
STRIDE=int(os.environ.get("OPENLET_EVAL_STRIDE","10"))
BATCH=4

def require(cond,msg):
    if not cond: raise RuntimeError(msg)

def load(path,label):
    require((path/"model.safetensors").is_file(),f"{label} model weights missing: {path}")
    require((path/"policy_preprocessor.json").is_file(),f"{label} preprocessor missing")
    require((path/"policy_postprocessor.json").is_file(),f"{label} postprocessor missing")
    print(f"[LOAD] {label}: {path}",flush=True)
    policy=ACTPolicy.from_pretrained(str(path),local_files_only=True,strict=True).to("cuda").eval()
    policy.reset()
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(path))
    return (policy,pre,post)

def evaluate_one(batch,model,pre,post,name):
    # Respect the same 0..1 visual convention as the official LeRobot train loop.
    raw={}
    for key,val in batch.items():
        if isinstance(val,torch.Tensor) and key.startswith("observation.images.") and val.dtype==torch.uint8:
            val=val.float().div_(255.0)
        raw[key]=val
    raw_action=raw["action"][:,0,:].float().to("cuda")
    if name=="OLD":
        for key in ("observation.images.head_cam_h","observation.images.wrist_cam_r"):
            img=raw[key]
            raw[key]=F.interpolate(img.float(),size=(480,848),mode="bilinear",align_corners=False)
    processed=pre(raw)
    if "action_is_pad" not in processed:
        raise RuntimeError("ACT chunk padding mask missing")
    valid=~processed["action_is_pad"][:,0].bool().to("cuda")
    with torch.inference_mode():
        pred_norm=model.select_action(processed)
        pred=post(pred_norm)
    if not isinstance(pred,torch.Tensor):
        pred=torch.as_tensor(pred,device="cuda")
    # Postprocessor may return CPU tensors even if policy predicts on CUDA.
    # Align both tensors before measuring action error in raw action units.
    pred=pred.to(device=raw_action.device, dtype=torch.float32)
    require(pred.shape==raw_action.shape, f"{name} output shape {pred.shape}, target {raw_action.shape}")
    errors=(pred-raw_action).abs().mean(dim=1)
    require(bool(torch.isfinite(errors).all()),f"{name}: nonfinite MAE")
    return errors.detach().cpu().tolist(),valid.cpu().tolist()

def main():
    t0=time.time()
    require(torch.cuda.is_available(),"CUDA not available")
    require(STRIDE>=1,"OPENLET_EVAL_STRIDE must be >=1")
    require(DATA.joinpath("meta/info.json").is_file(),"Dataset missing in training container")
    print("[EVAL] 50 held-out episodes (450-499), deterministic 1-in-"+str(STRIDE)+
          " frame sampling; first predicted action raw-unit MAE",flush=True)
    print("[EVAL] This is an offline action error, NOT a simulator score",flush=True)
    old=load(OLD,"OLD")
    new=load(NEW,"NEW")
    meta=LeRobotDatasetMetadata("OpenLET/task1_parts_pick_v3_500",root=DATA)
    delta=resolve_delta_timestamps(new[0].config,meta)
    ds=LeRobotDataset("OpenLET/task1_parts_pick_v3_500",root=DATA,
                      episodes=EPISODES,delta_timestamps=delta,
                      video_backend="pyav",return_uint8=True,tolerance_s=1e-4)
    print(f"[DATA] held-out frames={len(ds)}, fps={meta.fps}, sampled stride={STRIDE}",flush=True)
    indices=list(range(0,len(ds),STRIDE))
    if len(ds)-1 not in indices:indices.append(len(ds)-1)
    loader=DataLoader(Subset(ds,indices),batch_size=BATCH,shuffle=False,num_workers=2,
                      pin_memory=False,drop_last=False)
    sums={"OLD":0.0,"NEW":0.0};total={"OLD":0,"NEW":0}
    episodes=set(); per_ep={}
    for n,batch in enumerate(loader,1):
        ids=batch["episode_index"].tolist()
        for label,parts in (("OLD",old),("NEW",new)):
            errors,valid=evaluate_one(batch,*parts,label)
            for idx,error,ok in zip(ids,errors,valid):
                if not ok:continue
                idx=int(idx);episodes.add(idx)
                sums[label]+=float(error);total[label]+=1
                entry=per_ep.setdefault(idx,{"OLD":[0.0,0],"NEW":[0.0,0]})
                entry[label][0]+=float(error)
                entry[label][1]+=1
        if n==1 or n%50==0:
            print(f"[PROGRESS] batches={n}/{len(loader)} samples={total['NEW']} "
                  f"elapsed={time.time()-t0:.1f}s",flush=True)
    require(episodes==set(EPISODES),f"Coverage error: only {len(episodes)} episodes {sorted(episodes)}")
    require(total["OLD"]==total["NEW"] and total["NEW"]>0,"Invalid sample counts")
    means={x:sums[x]/total[x] for x in ("OLD","NEW")}
    per_episode={str(e):{label:per_ep[e][label][0]/per_ep[e][label][1]
                         for label in ("OLD","NEW")} for e in sorted(episodes)}
    new_wins=sum(v["NEW"]<v["OLD"] for v in per_episode.values())
    result={"metric":"mean absolute error across 21 raw action channels of next predicted action",
            "not_official_score":True,"episodes_held_out":EPISODES,
            "sampling_every_nth_frame":STRIDE,"sampled_frames":total["NEW"],
            "dataset_root":str(DATA),"old_checkpoint":str(OLD),"new_checkpoint":str(NEW),
            "old_resize":"bilinear 640x480 -> 848x480",
            "overall_mae":means,"new_wins_episode_count":new_wins,
            "per_episode_mae":per_episode,"elapsed_seconds":round(time.time()-t0,1)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("===== HELDOUT RESULT =====",flush=True)
    print(json.dumps({k:result[k] for k in ("sampled_frames","sampling_every_nth_frame","overall_mae",
                  "new_wins_episode_count","elapsed_seconds")},indent=2,ensure_ascii=False),flush=True)
    print("[PASS] 50 episodes covered. Report:",OUT,flush=True)
    print("[NOTE] Offline imitation MAE alone cannot establish Task1 success rate",flush=True)

if __name__=="__main__":
    main()
