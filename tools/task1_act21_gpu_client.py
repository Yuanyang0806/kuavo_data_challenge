#!/usr/bin/env python3
"""ACT10k client for Task1 21D ROS bridge (no ROS dependency in GPU container)."""
import argparse
import pickle
import socket
import struct
import time

import cv2
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

p=argparse.ArgumentParser()
p.add_argument("--host",required=True)
p.add_argument("--port",type=int,required=True)
p.add_argument("--checkpoint",required=True)
a=p.parse_args()
assert torch.cuda.is_available(),"CUDA unavailable"
print("[ACT] loading "+a.checkpoint,flush=True)
model=ACTPolicy.from_pretrained(a.checkpoint,local_files_only=True,strict=True).to("cuda").eval()
model.reset()
print("[ACT_CONFIG] chunk_size={} n_action_steps={}".format(
      model.config.chunk_size,model.config.n_action_steps),flush=True)
pre,post=make_pre_post_processors(policy_cfg=model.config,pretrained_path=a.checkpoint)
expected={"observation.state","observation.images.head_cam_h","observation.images.wrist_cam_r"}
assert set(model.config.input_features)==expected,model.config.input_features

def receive_exact(sock,n):
    buf=bytearray()
    while len(buf)<n:
        b=sock.recv(n-len(buf))
        if not b:raise EOFError("ROS bridge finished")
        buf.extend(b)
    return bytes(buf)
def receive(sock):
    n=struct.unpack("!I",receive_exact(sock,4))[0]
    if n>16000000:raise RuntimeError("oversized observation")
    return pickle.loads(receive_exact(sock,n))
def send(sock,obj):
    b=pickle.dumps(obj,protocol=4)
    sock.sendall(struct.pack("!I",len(b))+b)
def camera(blob):
    img=cv2.imdecode(np.frombuffer(blob,dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None:raise RuntimeError("compressed camera decode failed")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if img.shape[:2]!=(480,640):
        img=cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
    arr=np.ascontiguousarray(img)
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().div(255)

sock=socket.socket()
deadline=time.monotonic()+350
while True:
    try:
        sock.connect((a.host,a.port))
        break
    except OSError:
        if time.monotonic()>deadline:
            raise RuntimeError("cannot reach ROS bridge from training container")
        time.sleep(1)
sock.settimeout(35)
print("[ACT_CONNECTED] host={} port={}".format(a.host,a.port),flush=True)
count=0
lat_sum=0.0
lat_max=0.0
while True:
    try:
        obs=receive(sock)
    except EOFError:
        break
    state=torch.tensor(obs["state"],dtype=torch.float32).reshape(1,18)
    data={
        "observation.state":state,
        "observation.images.head_cam_h":camera(obs["head"]),
        "observation.images.wrist_cam_r":camera(obs["right"]),
    }
    tick=time.monotonic()
    with torch.inference_mode():
        predicted=model.select_action(pre(data))
        action=post(predicted)
    dt=time.monotonic()-tick
    lat_sum+=dt
    lat_max=max(lat_max,dt)
    if not isinstance(action,torch.Tensor):
        action=torch.as_tensor(action)
    action=action.detach().float().cpu().reshape(-1)
    if action.numel()!=21 or not bool(torch.isfinite(action).all()):
        raise RuntimeError("ACT returned invalid nonfinite/21D action")
    send(sock,action.tolist())
    count+=1
    if count==1 or count%25==0:
        print("[ACT_ACTION] step={} inference_ms={:.1f} avg_ms={:.1f}".format(
              count,dt*1000,lat_sum/count*1000),flush=True)
sock.close()
print("[ACT_DONE] steps={} avg_inference_ms={:.1f} max_inference_ms={:.1f}".format(
      count,lat_sum/max(1,count)*1000,lat_max*1000),flush=True)
