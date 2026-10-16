#!/usr/bin/env python3
"""One official scored Task1 episode with new ACT 21D, no repo overwrites.

Starts official model_entry, a separate temporary ROS bridge and LeTools GPU client.
Checks model/data layout, Docker mounts, and an idle simulator before acting.
Always stops the processes it started; preserves earlier model score files in report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_REV="847aedef5d6c4c26e8a7e4a73c2cbd4403e20439"
BASE_RAW="https://raw.githubusercontent.com/Yuanyang0806/kuavo_data_challenge/"+SCRIPT_REV+"/tools/"
OFFICIAL="kuavo-openlet-official-20261011"
TRAIN="letools-pi05"
BASE=Path.home()/"competitions/2026-openlet"
OFFICIAL_ROOT=BASE/"code/kuavo-ros-opensource-official-20261011"
TRAIN_ROOT=BASE/"code/LeTools-Learning"
DATA_META=BASE/"datasets/Embodied-Data-Manipulation-Challenge/sim/task1_parts_pick/meta/info.json"
CK_HOST=TRAIN_ROOT/"outputs/train/openlet_task1_v3_act_finetune_20261016_144053_model/checkpoints/010000/pretrained_model"
CK_DOCKER="/workspace/openlet/code/LeTools-Learning/outputs/train/openlet_task1_v3_act_finetune_20261016_144053_model/checkpoints/010000/pretrained_model"
SCORE_DIR=OFFICIAL_ROOT/"src/data_challenge_simulator/examples/model_scores/task1"

def say(*args):
    print(*args,flush=True)
def call(args,*,capture=False,timeout=60):
    return subprocess.run(args,check=True,text=True,capture_output=capture,timeout=timeout)
def sudo_docker(*args,**kw):
    return call(["sudo","docker",*map(str,args)],**kw)
def inspect(container):
    return json.loads(sudo_docker("inspect",container,capture=True).stdout)[0]
def die(msg):
    raise RuntimeError(msg)
def tail(path,n=15):
    if not path.is_file():return "<log not created>"
    content=path.read_text(errors="replace").replace("\r","\n")
    return "\n".join(content.splitlines()[-n:])
def docker_command(container,command,logfile,display=None):
    file=open(logfile,"w",buffering=1)
    cmd=["sudo","docker","exec"]
    if container==OFFICIAL:
        # scene1.sh passes these even for task-headless; MuJoCo's GlfwAdapter
        # still requires a functioning X/GLFW context.
        cmd += ["-e","ROBOT_VERSION=400062"]
        if display is not None:
            cmd += ["-e","DISPLAY="+display]
    cmd += [container,"bash","-lc",command]
    proc=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT)
    return proc,file

def find_working_display():
    # The MuJoCo C++ build creates a GLFW window even with MUJOCO_HEADLESS=1.
    # Test in the *same container* before starting ROS, to avoid 60s topic
    # waits and repeated auto-retries when DISPLAY was omitted.
    test_code = """
import ctypes, ctypes.util, os, sys
name = ctypes.util.find_library("glfw")
if not name:
    # If libglfw is statically linked, XOpenDisplay at least proves that
    # DISPLAY and X11 authentication are usable.
    lib = ctypes.CDLL("libX11.so.6")
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    ptr = lib.XOpenDisplay(None)
    if ptr:
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        lib.XCloseDisplay(ptr)
        sys.exit(0)
    sys.exit(2)
glfw = ctypes.CDLL(name)
glfw.glfwInit.restype = ctypes.c_int
if glfw.glfwInit() != 1:
    sys.exit(3)
try:
    glfw.glfwWindowHint.argtypes = [ctypes.c_int, ctypes.c_int]
    glfw.glfwWindowHint(0x00020004, 0)  # GLFW_VISIBLE=false
    glfw.glfwCreateWindow.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_char_p,
                                      ctypes.c_void_p,ctypes.c_void_p]
    glfw.glfwCreateWindow.restype = ctypes.c_void_p
    window = glfw.glfwCreateWindow(64,64,b"openlet-preflight",None,None)
    if not window:
        sys.exit(4)
    glfw.glfwDestroyWindow.argtypes = [ctypes.c_void_p]
    glfw.glfwDestroyWindow(window)
finally:
    glfw.glfwTerminate()
"""
    candidates = []
    # Prefer the exact host DISPLAY; fallback to well-known local X displays.
    for disp in (os.environ.get("DISPLAY"),":0",":1"):
        if disp and disp not in candidates:
            candidates.append(disp)
    # Also test the container's existing DISPLAY, if configured.
    env = sudo_docker("exec",OFFICIAL,"bash","-lc",'printf "%s" "$DISPLAY"',
                     capture=True).stdout.strip()
    if env and env not in candidates:
        candidates.append(env)
    for display in candidates:
        proc=subprocess.run(["sudo","docker","exec","-e","DISPLAY="+display,
                             OFFICIAL,"python3","-c",test_code],
                            capture_output=True,text=True,timeout=20,check=False)
        if proc.returncode==0:
            say("[PASS] container GLFW can initialize with DISPLAY="+display)
            return display
        say("[DISPLAY_CHECK] {} failed (exit={}); {}".format(
            display,proc.returncode,(proc.stderr or "").strip()[-200:]))
    die("MuJoCo requires X/GLFW even in headless mode. None of the DISPLAY "
        "candidates worked inside the official container. Check that the "
        "host X server is running and the container can access its socket "
        "and authentication; no simulator has been launched.")
def signal_model():
    subprocess.run(["sudo","docker","exec",OFFICIAL,"bash","-lc",
                    "pkill -INT -f '[m]odel_entry.py' || true"],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=12,check=False)
def cleanup(model,bridge,client):
    for proc in (bridge,client):
        if proc is not None and proc.poll() is None:
            proc.terminate()
    if model is not None and model.poll() is None:
        signal_model()
        try:model.wait(timeout=28)
        except subprocess.TimeoutExpired:
            subprocess.run(["sudo","docker","exec",OFFICIAL,"bash","-lc",
                            "pkill -TERM -f '[m]odel_entry.py' || true"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=12,check=False)
    for proc in (model,bridge,client):
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill()

def main():
    parser=argparse.ArgumentParser(description="Task1 ACT21 extended diagnostic")
    parser.add_argument("--steps",type=int,default=1200)
    parser.add_argument("--max-wall-seconds",type=float,default=180.0)
    args=parser.parse_args()
    if not 200<=args.steps<=2000:
        die("--steps must be in [200,2000]")
    if not 30<=args.max_wall_seconds<=300:
        die("--max-wall-seconds must be in [30,300]")
    stamp=time.strftime("%Y%m%d_%H%M%S")
    run_name="ACTV3_"+stamp
    report=Path.home()/"openlet-test-reports"/("act21_official_seed1_"+stamp)
    report.mkdir(parents=True,exist_ok=False)
    say("[REPORT]",report)
    say("[EVAL_WINDOW] steps={} max_wall_s={:.0f}".format(args.steps,args.max_wall_seconds))
    say("[NOTE] Extended diagnostic; official LeTools deployment default is 200 steps.")
    if not (CK_HOST/"model.safetensors").is_file():
        die("new 10k ACT weights missing")
    if not DATA_META.is_file():
        die("Task1 meta/info.json missing")
    features=json.loads(DATA_META.read_text())["features"]
    action_names=features["action"]["names"]["action_names"]
    state_names=features["observation.state"]["names"]["state_names"]
    right_arm=["zarm_r%d_link"%i for i in range(1,8)]
    right_hand=["right_sg100_%d"%i for i in range(1,12)]
    if action_names != right_arm+right_hand+["vx","vy","vyaw"]:
        die("action_names not right arm7+SG100x11+base3, refuse unsafe commands")
    if state_names != right_arm+right_hand:
        die("state_names not right arm7+SG100x11")
    task_meta=DATA_META.parent/"tasks.jsonl"
    tasks=[]
    if task_meta.is_file():
        with task_meta.open() as stream:
            for i,line in enumerate(stream):
                if i>=30:break
                try:tasks.append(json.loads(line))
                except json.JSONDecodeError:tasks.append({"unparsed":line[:160]})
    (report/"dataset_tasks_preview.json").write_text(json.dumps(
        {"dataset":str(DATA_META.parent.parent),"sample":tasks},
        indent=2,ensure_ascii=False))
    say("[TASK_META] stored {} task descriptions (if present)".format(len(tasks)))
    official=inspect(OFFICIAL);train=inspect(TRAIN)
    for item,name in ((official,OFFICIAL),(train,TRAIN)):
        if not item["State"]["Running"]:die(name+" not running")
    if official["HostConfig"]["NetworkMode"]!="host":
        die("official ROS container must use host networking")
    source=next((m["Source"] for m in official["Mounts"] if m["Destination"]=="/root/kuavo_ws"),None)
    if not source or Path(source).resolve()!=OFFICIAL_ROOT.resolve():
        die("official ROS container mount does not match independent official checkout")
    if not (OFFICIAL_ROOT/"src/data_challenge_simulator/scripts/model_entry.py").is_file():
        die("official model_entry.py missing")
    active=sudo_docker("exec",OFFICIAL,"bash","-lc",
           "pgrep -af '[m]odel_entry.py|[h]elperfunc.py|[n]odelet_mujoco|[t]ask1_act.*bridge.py' || true",
           capture=True).stdout
    active=[x for x in active.splitlines() if
            "pgrep -af" not in x and "bash -lc" not in x]
    if active:
        die("simulator already busy, refusing to interrupt:\n"+"\n".join(active[:10]))
    display=find_working_display()
    if train["HostConfig"]["NetworkMode"]=="host":
        host="127.0.0.1";bind="127.0.0.1"
    else:
        gateways=[n.get("Gateway") for n in train["NetworkSettings"]["Networks"].values()]
        gateways=[g for g in gateways if g and ":" not in g]
        if not gateways:die("training container has no reachable host IPv4 gateway")
        host=gateways[0];bind=host  # listen only on Docker gateway, not every host interface
    with socket.socket() as probe:
        probe.bind(("0.0.0.0",0))
        port=probe.getsockname()[1]
    say("[PASS] data 21/18, official volume, CUDA checkpoint, idle ROS")
    say("[NETWORK] bridge port={} client host={}".format(port,host))
    # model_entry resets the latest task1 model score files upon startup.
    if SCORE_DIR.is_dir():
        old=report/"previous_model_scores"
        old.mkdir()
        for name in ("score.json","score.txt","average.json","average.txt"):
            src=SCORE_DIR/name
            if src.is_file():shutil.copy2(src,old/name)

    with tempfile.TemporaryDirectory(prefix="openlet21_") as d:
        d=Path(d)
        for name in ("task1_act21_ros_bridge.py","task1_act21_gpu_client.py"):
            path=d/name
            call(["curl","-fsSL","--retry","3",BASE_RAW+name,"-o",str(path)],timeout=60)
            import py_compile
            py_compile.compile(str(path),doraise=True)
            dest=OFFICIAL if "bridge" in name else TRAIN
            sudo_docker("cp",path,dest+":/tmp/"+name)
    say("[PASS] separate scripts copied to /tmp in containers; original bridge untouched")

    model_log=report/"official_model_entry.log"
    bridge_log=report/"act21_bridge.log"
    client_log=report/"act21_gpu.log"
    model=bridge=client=None
    logs=[]
    telemetry_container="/tmp/openlet_task1_act21_{}_telemetry.jsonl".format(stamp)
    pid_file="/tmp/openlet_model_entry_{}.pid".format(stamp)
    run_error=None
    try:
        say("[START] official model_entry Task1 seed=1, headless")
        model_cmd=("source /opt/ros/noetic/setup.bash && "
                   "source /root/kuavo_ws/devel/setup.bash && "
                   "export PYTHONUNBUFFERED=1 KUAVO_LOG_SERVER=0 KUAVO_LEG_SERVICE=0 && "
                   "echo $ > "+shlex.quote(pid_file)+" && "
                   "exec python3 -u /root/kuavo_ws/src/data_challenge_simulator/scripts/model_entry.py "
                   "--task-id 1 --seed 1 --headless --model-name "+shlex.quote(run_name))
        model,f=docker_command(OFFICIAL,model_cmd,model_log,display=display);logs.append(f)
        time.sleep(7)
        if model.poll() is not None:die("model_entry exited during startup:\n"+tail(model_log))
        say("[START] temporary ROS bridge")
        bridge_cmd=("source /opt/ros/noetic/setup.bash && "
                    "source /root/kuavo_ws/devel/setup.bash && "
                    "exec python3 -u /tmp/task1_act21_ros_bridge.py "
                    "--bind {} --port {} --steps {} --max-wall-seconds {:.1f} --telemetry {}".format(
                        shlex.quote(bind),port,args.steps,args.max_wall_seconds,
                        shlex.quote(telemetry_container)))
        bridge,f=docker_command(OFFICIAL,bridge_cmd,bridge_log,display=display);logs.append(f)
        time.sleep(3)
        if bridge.poll() is not None:die("bridge exited during startup:\n"+tail(bridge_log))
        say("[START] LeTools new ACT GPU inference")
        client_cmd="export PYTHONPATH=/workspace/openlet/code/LeTools-Learning/third_party/lerobot/src:/workspace/openlet/code/LeTools-Learning:$PYTHONPATH; export PYTHONUNBUFFERED=1; exec python3 -u /tmp/task1_act21_gpu_client.py --host {} --port {} --checkpoint {}".format(shlex.quote(host),port,shlex.quote(CK_DOCKER))
        client,f=docker_command(TRAIN,client_cmd,client_log);logs.append(f)
        deadline=time.monotonic()+args.max_wall_seconds+400
        last=""
        while time.monotonic()<deadline:
            time.sleep(5)
            # model_entry automatically retries another random seed when
            # initial topic checks fail; abort instead of running a wrong seed.
            if "[ERROR] model entry: task 1 seed 1 failed before teardown" in model_log.read_text(errors="replace"):
                say("[STARTUP_ABORT] official seed 1 did not initialize; no score")
                break
            if bridge.poll() is not None or client.poll() is not None or model.poll() is not None:
                break
            msgs=[line for line in tail(bridge_log,30).splitlines()
                  if any(k in line for k in ("[BRIDGE_READY]","[OFFICIAL_SCORING_STARTED]","[ACTION]","[BRIDGE_DONE]"))]
            if msgs and msgs[-1]!=last:
                last=msgs[-1];say(last)
        else:
            say("[TIMEOUT] maximum 650s exceeded")
        say("[EXIT] model={} bridge={} client={}".format(model.poll(),bridge.poll(),client.poll()))
    finally:
        cleanup(model,bridge,client)
        for f in logs:f.close()
        time.sleep(2)

    for title,path in (("OFFICIAL",model_log),("BRIDGE",bridge_log),("CLIENT",client_log)):
        say("\n===== "+title+" LOG =====")
        say(tail(path,17))
    score=SCORE_DIR/"score.json"
    if score.is_file():
        obj=json.loads(score.read_text())
        run=obj.get("model_run") or {}
        if run.get("model_name")==run_name and int(run.get("seed",-1))==1 and isinstance(obj.get("total"),(int,float)):
            shutil.copy2(score,report/"official_score.json")
            say("\n[OFFICIAL_MODEL_SCORE]",json.dumps({
                "score":obj["total"],"seed":1,"model":run_name,
                "reason":run.get("finish_reason"),
                "score_file":str(report/"official_score.json")
            },ensure_ascii=False))
            return
    say("\n[NO_VALID_SCORE] The official scorer has not finalized this run; do not invent score.")
    say("[REPORT]",report)
    raise SystemExit(2)

if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:
        say("[INTERRUPTED] attempt graceful stop")
        raise SystemExit(130)
    except Exception as exc:
        say("[ERROR] {}: {}".format(type(exc).__name__,exc))
        raise SystemExit(1)
