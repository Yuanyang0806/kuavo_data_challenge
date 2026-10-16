#!/usr/bin/env python3
"""One-shot launcher: OpenLET Task1 V3 500 episodes -> isolated ACT fine-tune."""
from pathlib import Path
import json, os, shlex, subprocess, sys, time

HOME=Path.home()
BASE=HOME/"competitions/2026-openlet"
DATA=BASE/"datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick"
LOCAL_LET=BASE/"code/LeTools-Learning"
C="letools-pi05"
R=Path("/workspace/openlet/code/LeTools-Learning")
CK=R/"outputs/train/openlet_task1_act_competition_20261008_123402/checkpoints/020000/pretrained_model"

def run(args, **opts):
    result=subprocess.run(args,**opts)
    if result.returncode:
        raise RuntimeError("Command failed with code %s: %s"%(result.returncode," ".join(map(str,args[:5]))))
    return result

def dex(*a, **kw):
    return run(["sudo","docker","exec","-i",C,*map(str,a)],**kw)

def die(s):
    raise SystemExit("[STOP] "+s)

print("=== 官方Task1新版500条数据：ACT训练 ===",flush=True)
if not (DATA/"meta/info.json").exists():die("新版数据不存在")
meta=json.loads((DATA/"meta/info.json").read_text())
if meta.get("total_episodes")!=500:die("不是500条新版数据")
expected={"observation.state":[18],"action":[21],
          "observation.images.head_cam_h":[3,480,640],
          "observation.images.wrist_cam_r":[3,480,640]}
for k,shape in expected.items():
    if meta.get("features",{}).get(k,{}).get("shape")!=shape:
        die("数据特征不一致: "+k)
media=[p for p in DATA.rglob("*") if p.suffix in (".mp4",".parquet") and "20261015" not in p.parts and p.is_file()]
if not media or any(p.stat().st_size<512 and p.open("rb").read(64).startswith(b"version https://git-lfs.github.com/spec/v1") for p in media):
    die("数据尚未全部下载")
print("[PASS] 500条数据、18维状态、21维动作、两路640x480相机",flush=True)
subprocess.run(["sudo","docker","start",C],stdout=subprocess.DEVNULL,check=False)
dex("test","-f",CK/"config.json")
dex("test","-f",R/"configs/train/lerobot/total/act_total.yaml")
dex("python3","-c","import torch,yaml;assert torch.cuda.is_available();print('[PASS] CUDA:',torch.cuda.get_device_name(0))")
# Prevent concurrent training on same GPU.
check=dex("bash","-lc","ps -eo pid,args | grep '[p]ython.*lerobot.scripts.lerobot_train' || true",
          text=True,capture_output=True)
if check.stdout.strip():
    die("发现已有LeRobot训练，请避免重复占用显存:\n"+check.stdout[:700])

# Resolve host dataset mount, otherwise copy only new data to the container.
mounts=json.loads(run(["sudo","docker","inspect","-f","{{json .Mounts}}",C],capture_output=True,text=True).stdout)
container_data=None
for m in mounts:
    try:rel=DATA.resolve().relative_to(Path(m["Source"]).resolve())
    except ValueError:continue
    candidate=Path(m["Destination"])/rel
    if subprocess.run(["sudo","docker","exec",C,"test","-f",str(candidate/"meta/info.json")],stdout=subprocess.DEVNULL).returncode==0:
        container_data=candidate
        break
if container_data is None:
    container_data=Path("/workspace/openlet/datasets/task1_parts_pick_v3_500")
    present=subprocess.run(["sudo","docker","exec",C,"test","-f",str(container_data/"meta/info.json")],
                           stdout=subprocess.DEVNULL).returncode==0
    if not present:
        print("[DATA] 正将已有的新版数据复制到训练容器（约6GB），不重新下载",flush=True)
        dex("mkdir","-p",container_data)
        producer=subprocess.Popen(["tar","-C",str(DATA),"--exclude=20261015","-cf","-","."],stdout=subprocess.PIPE)
        consumer=subprocess.Popen(["sudo","docker","exec","-i",C,"tar","-C",str(container_data),"-xf","-"],stdin=producer.stdout)
        producer.stdout.close()
        codes=(producer.wait(),consumer.wait())
        if codes!=(0,0):die("复制数据失败: "+str(codes))
print("[PASS] 训练容器数据目录:",container_data,flush=True)

stamp=time.strftime("%Y%m%d_%H%M%S")
run_id="openlet_task1_v3_act_finetune_"+stamp
output=R/"outputs/train"/run_id
prepare=r'''
from pathlib import Path
import yaml,json,sys
repo=Path(sys.argv[1]);data=Path(sys.argv[2]);out=Path(sys.argv[3]);ck=Path(sys.argv[4])
original=yaml.safe_load((repo/"configs/train/lerobot/total/act_total.yaml").read_text())
cfg=dict(original.pop("training"));cfg.update(original)
cfg.update(output_dir=str(out),job_name=out.name,resume=False,steps=10000,
           batch_size=4,num_workers=2,prefetch_factor=2,persistent_workers=False,
           save_checkpoint=True,save_freq=2000,log_freq=50,eval_freq=0,seed=20261016)
cfg["dataset"].update(repo_id="OpenLET/task1_parts_pick_v3_500",root=str(data),
                      episodes=list(range(450)),video_backend="pyav",streaming=False)
cfg["wandb"]["enable"]=False
cfg["policy"].update(pretrained_path=str(ck),type="act",device="cuda",use_amp=False,
                      push_to_hub=False,chunk_size=100,n_action_steps=1)
cfg["policy"]["input_features"]={
 "observation.state":{"type":"STATE","shape":[18]},
 "observation.images.head_cam_h":{"type":"VISUAL","shape":[3,480,640]},
 "observation.images.wrist_cam_r":{"type":"VISUAL","shape":[3,480,640]}}
cfg["policy"]["output_features"]={"action":{"type":"ACTION","shape":[21]}}
if isinstance(cfg.get("scheduler"),dict):
 cfg["scheduler"]["num_decay_steps"]=10000
 cfg["scheduler"]["num_warmup_steps"]=250
out.mkdir(parents=True,exist_ok=False)
(out/"train_config_requested.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
(out/"dataset_split.json").write_text(json.dumps({"train_episodes":[0,449],"held_out_episodes":[450,499]},indent=2))
print("[PASS] 新微调配置完成",out)
'''
dex("python3","-",R,container_data,output,CK,input=prepare,text=True)

q=shlex.quote
script=("cd "+q(str(R))+"; export PYTHONPATH="+q(str(R/"third_party/lerobot/src"))+":"+
        q(str(R))+":$PYTHONPATH; export PYTHONUNBUFFERED=1; "+
        "python3 -u -m lerobot.scripts.lerobot_train --config_path="+
        q(str(output/"train_config_requested.json"))+" > "+q(str(output/"train.log"))+
        " 2>&1; rc=$?; echo $rc > "+q(str(output/"exit_status.txt"))+"; exit $rc")
run(["sudo","docker","exec","-d",C,"bash","-lc",script])
print("[LAUNCHED] 新数据ACT微调进程已提交",flush=True)
print("RUN:",run_id)
print("DATA: 450 train / 50 held out")
print("OLD CHECKPOINT: preserved")
print("TRAIN LOG (host):",LOCAL_LET/"outputs/train"/run_id/"train.log")
time.sleep(12)
check=dex("bash","-lc","tail -n 18 "+q(str(output/"train.log"))+
          "; if test -f "+q(str(output/"exit_status.txt"))+
          "; then echo TRAIN_EXIT=$(cat "+q(str(output/"exit_status.txt"))+"); fi",
          capture_output=True,text=True)
print("===== 当前训练日志 =====",flush=True)
print(check.stdout[-4200:])
if "TRAIN_EXIT=" in check.stdout and "TRAIN_EXIT=0" not in check.stdout:
    die("训练进程已退出，请查看上述日志")
print("[NEXT] 新训练已经提交。正式ACT仿真成绩还需要ROS策略执行桥接。",flush=True)
