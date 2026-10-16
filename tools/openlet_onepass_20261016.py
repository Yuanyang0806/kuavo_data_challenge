from pathlib import Path
import os, sys, json, re, shutil, subprocess, time, signal, traceback

HOME = Path.home()
BASE = HOME / 'competitions/2026-openlet'
NEW = BASE / 'code/kuavo-ros-opensource-official-20261011'
DATASET = BASE / 'datasets/Embodied-Data-Manipulation-Challenge'
DATA = DATASET / 'sim/task1_parts_pick'
FILE = NEW / 'src/data_challenge_simulator/examples/task1.py'
OUT = HOME / 'openlet-test-reports'
OUT.mkdir(exist_ok=True)
STAMP = time.strftime('%Y%m%d_%H%M%S')
CONTAINER = 'kuavo-openlet-official-20261011'
TRAIN = 'letools-pi05'
results = {'脚本回归': '未执行', '官方新数据': '未执行', '旧ACT前向': '未执行', 'ACT官方分数': '未评测（需要ROS动作桥接）'}

def call(args, *, timeout=None, log=None, input=None, env=None):
    if log:
        with open(log, 'w') as f:
            return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                  input=input, text=True, env=env).returncode
    return subprocess.run(args, input=input, text=True, timeout=timeout, env=env, check=False).returncode

def load_score(path, t):
    try:
        if path.stat().st_mtime < t: return None
        return json.loads(path.read_text()).get('total')
    except Exception: return None

PATCH = '''            # OPENLET_RECENTER_V2: fix pre-grasp cylinder displacement only.
            if name == "cylinder_1":
                for recenter_attempt in range(2):
                    latest = poses.get_position(name)
                    drift = math.hypot(latest[0]-object_position[0],
                                       latest[1]-object_position[1])
                    if drift <= 0.014:
                        break
                    if drift > 0.075:
                        raise RuntimeError("{} displaced {:.3f}m".format(name,drift))
                    print("[OPENLET_RECENTER] {} attempt {} drift {:.4f}m".format(
                        name,recenter_attempt+1,drift))
                    revised = [latest[k]-grasp_bias_world[k] for k in range(3)]
                    opts = dict(point_in_r7=scene_ik.CLOSED_GRASP_CENTER_IN_R7,
                                compensate_source_grasp=True)
                    move_right_tcp(name+"_recenter_retract",
                        [grasp_center_world[0],grasp_center_world[1],grasp_center_world[2]+0.10],
                        task_tcp_quaternion_world,trajectory_points=30,**opts)
                    move_right_tcp(name+"_recenter_above",
                        [revised[0],revised[1],revised[2]+0.10],
                        task_tcp_quaternion_world,trajectory_points=40,**opts)
                    move_right_tcp(name+"_recenter_descend",revised,
                        task_tcp_quaternion_world,trajectory_points=40,**opts)
                    object_position = latest
                    grasp_center_world = revised
                    time.sleep(0.25)
                else:
                    current = poses.get_position(name)
                    if math.hypot(current[0]-object_position[0],
                                  current[1]-object_position[1])>0.014:
                        raise RuntimeError("{} recenter failed".format(name))
'''

print('\n=== 1/3 修复 Seed4 并自动回归 ===', flush=True)
if not FILE.is_file():
    results['脚本回归']='失败：新版 task1.py 不存在'
else:
    original = FILE.read_bytes()
    backup = OUT / f'task1_before_{STAMP}.py'
    shutil.copy2(FILE,backup)
    changed = False
    passed = False
    try:
        src = original.decode('utf-8')
        anchor = '            pre_close_object_position = poses.get_position(name)\n'
        if 'OPENLET_RECENTER_V2' in src:
            print('[INFO] 检测到已安装补丁，直接测试', flush=True)
        elif src.count(anchor)!=1:
            raise RuntimeError('源码插入位置不匹配，已停止修改')
        else:
            new_src = src.replace(anchor,PATCH+anchor,1)
            compile(new_src,str(FILE),'exec')
            FILE.write_text(new_src)
            changed = True
            print('[PASS] 重定位补丁已安装（只影响 cylinder_1）', flush=True)
        call(['xhost','+SI:localuser:root'],timeout=10,log=OUT/f'xhost_{STAMP}.log') if shutil.which('xhost') else None
        for seed in (4,1,2,3):
            score = NEW / f'src/data_challenge_simulator/examples/scores/score_task1_round{seed}.json'
            log = OUT / f'onerun_seed{seed}_{STAMP}.log'
            if score.exists(): shutil.copy2(score,OUT/f'score_before_seed{seed}_{STAMP}.json')
            start=int(time.time())
            print(f'[RUN] Seed {seed}（上限 300 秒）',flush=True)
            envargs=['DISPLAY='+os.environ.get('DISPLAY',':0'), 'SCENE1_CONTAINER='+CONTAINER]
            try:
                rc=call(['timeout','--signal=INT','--kill-after=20s','300s',
                         'sudo','env',*envargs,'bash',str(NEW/'scene1.sh'),'task-headless',str(seed)],
                        timeout=340,log=log)
            except subprocess.TimeoutExpired: rc=124
            txt=log.read_text(errors='replace') if log.exists() else ''
            value=load_score(score,start)
            ok=(rc==0 and '任务成功' in txt and isinstance(value,(int,float)) and value==100)
            print(f'[TEST] Seed {seed}: score={value} success={ok} exit={rc}',flush=True)
            if not ok:
                print('[LOG]', '\n'.join([s for s in txt.splitlines() if any(k in s for k in
                    ('[OPENLET_RECENTER]','Task 1 V2 failed','任务失败','得分'))][-8:]),flush=True)
                raise RuntimeError(f'Seed {seed} 回归未满分')
        passed=True
        results['脚本回归']='通过：Seed 1–4 均得 100，保留补丁'
    except Exception as e:
        results['脚本回归']='失败：'+str(e)
    finally:
        if not passed and changed:
            shutil.copy2(backup,FILE)
            print('[ROLLBACK] 原任务脚本自动恢复',flush=True)

print('\n=== 2/3 旧 ACT 20k checkpoint CUDA 前向 ===',flush=True)
smoke=r'''import sys,json
from pathlib import Path
root=Path('/workspace/openlet/code/LeTools-Learning')
sys.path.insert(0,str(root/'third_party/lerobot/src'))
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
ck=root/'outputs/train/openlet_task1_act_competition_20261008_123402/checkpoints/020000/pretrained_model'
assert ck.is_dir(),f'checkpoint missing: {ck}'
assert torch.cuda.is_available(),'CUDA unavailable'
cfg=json.loads((ck/'config.json').read_text())
model=ACTPolicy.from_pretrained(str(ck),local_files_only=True,strict=True).to('cuda').eval()
model.reset()
obs={}
for k,s in cfg['input_features'].items():
    dims=[1]+list(s['shape'])
    obs[k]=torch.rand(*dims,device='cuda') if s['type']=='VISUAL' else torch.zeros(*dims,device='cuda')
with torch.inference_mode():action=model.select_action(obs)
assert tuple(action.shape)==(1,21),tuple(action.shape)
assert bool(torch.isfinite(action).all()),'non finite action'
print('[PASS] ACT checkpoint GPU_FORWARD_OK shape=',tuple(action.shape))
print('[NOTE] synthetic observation != official model score')
'''
try:
    if call(['sudo','docker','start','letools-pi05'],timeout=45,log=OUT/f'train_start_{STAMP}.log'):
        raise RuntimeError('训练容器启动失败')
    log=OUT/f'act_smoke_{STAMP}.log'
    rc=call(['sudo','docker','exec','-i','letools-pi05','python','-'],input=smoke,timeout=240,log=None)
    results['旧ACT前向']='通过：权重加载和CUDA动作输出正常' if rc==0 else '失败：GPU前向失败'
except Exception as e:
    results['旧ACT前向']='失败：'+str(e)

print('\n=== 3/3 新数据下载和兼容性检查 ===',flush=True)
if not (DATASET/'.git').is_dir() or not (DATA/'meta/info.json').is_file():
    results['官方新数据']='失败：数据目录缺失（原克隆保留）'
elif call(['git','lfs','version'],log=OUT/f'lfs_version_{STAMP}.log')!=0:
    results['官方新数据']='失败：未安装 git-lfs'
else:
    info=json.loads((DATA/'meta/info.json').read_text())
    pointer_size=0
    for f in DATA.rglob('*'):
        if not f.is_file() or '20261015' in f.parts or f.stat().st_size>512: continue
        content=f.read_bytes()[:400].decode('ascii','ignore')
        if content.startswith('version https://git-lfs.github.com/spec/v1'):
            m=re.search(r'(?m)^size (\d+)$',content)
            if m:pointer_size+=int(m.group(1))
    free=shutil.disk_usage(DATA).free
    print(f'[DATA] 官方元数据 episodes={info.get("total_episodes")} 待下载约 {pointer_size/2**30:.2f}GiB 可用 {free/2**30:.2f}GiB',flush=True)
    if pointer_size>0 and free < pointer_size*2.1 + 4*(1024**3):
        results['官方新数据']='跳过：硬盘空间不足（避免下载中断或磁盘耗尽）'
    else:
        log=OUT/f'dataset_lfs_{STAMP}.log'
        print('[DATA] 下载日志: '+str(log),flush=True)
        print('[DATA] 下载正确目录 sim/task1_parts_pick/，排除 20261015 旧版',flush=True)
        try:
            rc=call(['git','-C',str(DATASET),'lfs','pull',
                '--include=sim/task1_parts_pick/**',
                '--exclude=sim/task1_parts_pick/20261015/**'],timeout=10800,log=log)
        except subprocess.TimeoutExpired:rc=124
        files=[f for f in DATA.rglob('*') if f.is_file() and '20261015' not in f.parts and f.suffix in ('.mp4','.parquet')]
        pointers=[f for f in files if f.stat().st_size<512 and
                  f.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1')]
        feat=info.get('features',{})
        old={'observation.state':[18],'action':[21],
             'observation.images.head_cam_h':[3,480,848],
             'observation.images.wrist_cam_r':[3,480,848]}
        comp={k:{'new':feat.get(k,{}).get('shape'),'old':v,
                 'compatible_shape':(feat.get(k,{}).get('shape') in [v,[480,848,3]] if 'images' in k
                                     else feat.get(k,{}).get('shape')==v)} for k,v in old.items()}
        print('[DATA] files={} pointers={} lfs_exit={}'.format(len(files),len(pointers),rc),flush=True)
        print('[DATA] ACT特征比较:',json.dumps(comp,ensure_ascii=False),flush=True)
        (OUT/f'data_compat_{STAMP}.json').write_text(json.dumps({'episodes':info.get('total_episodes'),
            'lfs_exit':rc,'pointers':len(pointers),'features':comp},ensure_ascii=False,indent=2))
        results['官方新数据']=('通过：500条元数据、文件完整' if rc==0 and len(pointers)==0 and files and
                 info.get('total_episodes')==500 else '需处理：下载/元数据不完整，见日志')
        if any(not z['compatible_shape'] for z in comp.values()):
            results['官方新数据']+='；与旧ACT维度存在差异，不自动混训'

print('\n================ 一次运行结果 ================')
for k,v in results.items():print(f'{k}：{v}')
print('本地日志目录：',OUT)
print('无需上传文件；旧 ACT 的在线分数仍需连接真实 ROS 推理桥接。')
