#!/usr/bin/env python3
"""Diagnose expert motion vs new ACT on 50 held-out Task1 V3 episodes.

Read-only deterministic 1-in-10 frame sample. Does not run simulator or train.
Reports moving-frame base accuracy separately from idle-dominated 21D MAE.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT=Path("/workspace/openlet")
LE=ROOT/"code/LeTools-Learning"
DS=ROOT/"datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick"
CK=LE/"outputs/train/openlet_task1_v3_act_finetune_20261016_144053_model/checkpoints/010000/pretrained_model"
OUT=LE/"outputs/eval/task1_v3_act_motion_diagnostic.json"
EP=list(range(450,500))
STRIDE=10
THRESHOLD=0.03

def main():
    start=time.time()
    assert (DS/"meta/info.json").is_file()
    assert (CK/"model.safetensors").is_file()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    print("[MOTION] 50 held-out episodes (450-499), every 10th frame",flush=True)
    print("[MOTION] compare expert vs ACT, report base motion at threshold 0.03",flush=True)
    model=ACTPolicy.from_pretrained(str(CK),local_files_only=True,strict=True).to("cuda").eval()
    pre,post=make_pre_post_processors(policy_cfg=model.config,pretrained_path=str(CK))
    meta=LeRobotDatasetMetadata("OpenLET/task1_parts_pick_v3_500",root=DS)
    ds=LeRobotDataset("OpenLET/task1_parts_pick_v3_500",root=DS,
         episodes=EP,delta_timestamps=resolve_delta_timestamps(model.config,meta),
         video_backend="pyav",return_uint8=True,tolerance_s=1e-4)
    ids=list(range(0,len(ds),STRIDE))
    if len(ds)-1 not in ids:ids.append(len(ds)-1)
    loader=DataLoader(Subset(ds,ids),batch_size=4,shuffle=False,num_workers=2)
    truth=[];prediction=[];state=[];epids=[]
    for n,batch in enumerate(loader,1):
        clean={}
        for k,v in batch.items():
            if isinstance(v,torch.Tensor) and k.startswith("observation.images.") and v.dtype==torch.uint8:
                v=v.float().div(255.0)
            clean[k]=v
        if "action_is_pad" in batch:
            valid=(~batch["action_is_pad"][:,0].bool()).cpu()
        else:
            raise RuntimeError("padding mask missing; cannot validate")
        with torch.inference_mode():
            action=post(model.select_action(pre(clean)))
        if not isinstance(action,torch.Tensor):action=torch.as_tensor(action)
        action=action.detach().float().cpu()
        target=batch["action"][:,0,:].float().cpu()
        if action.shape!=target.shape or action.shape[-1]!=21:
            raise RuntimeError("action shape mismatch")
        if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(target).all()):
            raise RuntimeError("nonfinite action")
        truth.append(target[valid])
        prediction.append(action[valid])
        state.append(batch["observation.state"].float().cpu()[valid])
        epids.extend(batch["episode_index"].cpu()[valid].tolist())
        if n==1 or n%150==0:
            print("[PROGRESS] {}/{} batches".format(n,len(loader)),flush=True)
    y=torch.cat(truth)
    p=torch.cat(prediction)
    x=torch.cat(state)
    assert y.shape[0]==len(epids)>0
    assert len(set(epids))==50,sorted(set(epids))
    base_true=y[:,18:21]
    base_pred=p[:,18:21]
    base_norm_true=base_true.norm(dim=1)
    base_norm_pred=base_pred.norm(dim=1)
    moving=base_norm_true>THRESHOLD
    pred_moving=base_norm_pred>THRESHOLD
    groups={"right_arm":(0,7),"right_sg100":(7,18),"base_vel":(18,21)}
    group_mae={k:round(float((p[:,s:e]-y[:,s:e]).abs().mean()),7)
               for k,(s,e) in groups.items()}
    active_mae=None
    active_zero_mae=None
    active_direction_cosine=None
    if bool(moving.any()):
        active_mae=float((base_pred[moving]-base_true[moving]).abs().mean())
        active_zero_mae=float(base_true[moving].abs().mean())
        active_direction_cosine=float(torch.nn.functional.cosine_similarity(
           base_pred[moving],base_true[moving],dim=1,eps=1e-9).mean())
    per_ep=defaultdict(lambda:{"total":0,"expert_moving":0,"pred_moving":0})
    for i,ep in enumerate(epids):
        k=int(ep)
        per_ep[k]["total"]+=1
        per_ep[k]["expert_moving"]+=int(moving[i])
        per_ep[k]["pred_moving"]+=int(pred_moving[i])
    moving_episodes=[k for k,v in per_ep.items() if v["expert_moving"]>0]
    report={
      "kind":"offline_diagnostic_not_official_score",
      "model":str(CK),
      "heldout_episodes":"450-499",
      "sample_every_nth_frame":STRIDE,
      "frame_count":int(y.shape[0]),
      "moving_threshold_l2_base_velocity":THRESHOLD,
      "expert_base_moving_frames":int(moving.sum()),
      "expert_base_moving_fraction":round(float(moving.float().mean()),6),
      "pred_base_moving_frames":int(pred_moving.sum()),
      "pred_base_moving_fraction":round(float(pred_moving.float().mean()),6),
      "expert_base_moving_episodes":len(moving_episodes),
      "base_ground_truth_mean_speed":round(float(base_norm_true.mean()),7),
      "base_model_mean_speed":round(float(base_norm_pred.mean()),7),
      "base_model_mae":round(float((base_pred-base_true).abs().mean()),7),
      "base_zero_prediction_mae":round(float(base_true.abs().mean()),7),
      "base_model_mae_on_expert_moving_frames":round(active_mae,7) if active_mae is not None else None,
      "base_zero_mae_on_expert_moving_frames":round(active_zero_mae,7) if active_zero_mae is not None else None,
      "moving_frame_base_direction_cosine":round(active_direction_cosine,5) if active_direction_cosine is not None else None,
      "group_mae":group_mae,
      "arm_target_minus_current_mean_abs":round(float((y[:,:7]-x[:,:7]).abs().mean()),7),
      "arm_model_minus_current_mean_abs":round(float((p[:,:7]-x[:,:7]).abs().mean()),7),
      "sg100_target_minus_current_mean_abs":round(float((y[:,7:18]-x[:,7:18]).abs().mean()),7),
      "sg100_model_minus_current_mean_abs":round(float((p[:,7:18]-x[:,7:18]).abs().mean()),7),
      "episodes_with_expert_base_motion":sorted(moving_episodes),
      "elapsed_seconds":round(time.time()-start,1)
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("===== MOTION DIAGNOSTIC RESULT =====",flush=True)
    print(json.dumps({k:v for k,v in report.items()
      if k not in ("model","episodes_with_expert_base_motion")},
      ensure_ascii=False,indent=2),flush=True)
    print("[PASS] report:",OUT,flush=True)
    if report["expert_base_moving_frames"]==0:
        print("[INTERPRET] Held-out expert base moves not visible in 1Hz sampling.",flush=True)
    elif report["pred_base_moving_fraction"]<.1*report["expert_base_moving_fraction"]:
        print("[INTERPRET] Model base motion is far less frequent than expert base motion.",flush=True)
    else:
        print("[INTERPRET] No categorical base-action collapse detected; inspect ROS closed-loop dynamics.",flush=True)

if __name__=="__main__":
    main()
