#!/usr/bin/env python3
"""Temporary 21D right-arm SG100+base adapter for official OpenLET model_entry."""
import argparse, json, math, pickle, socket, struct, threading, time
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from kuavo_msgs.msg import sensorsData, SG100HandState, SG100HandCommand

p=argparse.ArgumentParser()
p.add_argument("--port",type=int,required=True)
p.add_argument("--bind",default="127.0.0.1")
p.add_argument("--steps",type=int,default=1200)
p.add_argument("--max-wall-seconds",type=float,default=180.0)
p.add_argument("--telemetry",required=True)
args=p.parse_args()
rospy.init_node("openlet_task1_act21_bridge",anonymous=True)
lock=threading.Lock()
obs={k:None for k in ("head","right","sensor","hand")}
t_seen={k:0.0 for k in obs}
success=False
base_pose=None

def update(key,obj):
    with lock:
        obs[key]=obj
        t_seen[key]=time.monotonic()
def success_cb(m):
    global success
    if m.data:success=True
def base_cb(m):
    global base_pose
    with lock:
        base_pose=(float(m.pose.position.x),float(m.pose.position.y),float(m.pose.position.z))
rospy.Subscriber("/cam_h/color/image_raw/compressed",CompressedImage,lambda m:update("head",bytes(m.data)),queue_size=1)
rospy.Subscriber("/cam_r/color/image_raw/compressed",CompressedImage,lambda m:update("right",bytes(m.data)),queue_size=1)
rospy.Subscriber("/sensors_data_raw",sensorsData,lambda m:update("sensor",m),queue_size=1)
rospy.Subscriber("/sg100_hand_state",SG100HandState,lambda m:update("hand",m),queue_size=1)
rospy.Subscriber("/simulator/success",Bool,success_cb,queue_size=1)
rospy.Subscriber("/mujoco/base_link/pose",PoseStamped,base_cb,queue_size=1)
arms=rospy.Publisher("/kuavo_arm_traj",JointState,queue_size=1)
hands=rospy.Publisher("/sg100_hand_command",SG100HandCommand,queue_size=1)
base=rospy.Publisher("/cmd_vel",Twist,queue_size=1)

def get_observation():
    with lock:
        if any(obs[k] is None or time.monotonic()-t_seen[k]>3 for k in obs):
            return None
        return dict(obs)
def recv_exact(conn,n):
    b=bytearray()
    while len(b)<n:
        v=conn.recv(n-len(b))
        if not v:raise EOFError("ACT client disconnected")
        b.extend(v)
    return bytes(b)
def recv_obj(conn):
    n=struct.unpack("!I",recv_exact(conn,4))[0]
    if n>1048576:raise ValueError("bad action packet size")
    return pickle.loads(recv_exact(conn,n))
def send_obj(conn,data):
    blob=pickle.dumps(data,protocol=4)
    conn.sendall(struct.pack("!I",len(blob))+blob)

def wait_ready():
    deadline=time.monotonic()+250
    while not rospy.is_shutdown() and time.monotonic()<deadline:
        try:
            r=rospy.wait_for_message("/model_simulator/ready",Bool,timeout=1)
            a=rospy.wait_for_message("/model_simulator/accept_commands",Bool,timeout=1)
            if r.data and a.data and get_observation() is not None:
                return
        except rospy.ROSException:
            pass
    raise RuntimeError("model_entry did not publish ready/accept within 250s")

wait_ready()
print("[BRIDGE_READY] official model handoff completed",flush=True)
srv=socket.socket()
srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
srv.bind((args.bind,args.port))
srv.listen(1)
srv.settimeout(240)
print("[BRIDGE_SOCKET] listening {}:{}".format(args.bind,args.port),flush=True)
conn,addr=srv.accept()
conn.settimeout(30)
print("[BRIDGE_CLIENT] connected from {}".format(addr),flush=True)
rospy.wait_for_service("/simulator/start",timeout=25)
started=rospy.ServiceProxy("/simulator/start",Trigger)()
if not started.success:
    raise RuntimeError("official /simulator/start rejected: "+started.message)
print("[OFFICIAL_SCORING_STARTED]",started.message,flush=True)
count=0
t0=time.monotonic()
logs=open(args.telemetry,"w",buffering=1)
terminal_reason="max_steps"
previous_action=None
try:
    rate=rospy.Rate(10)
    while not rospy.is_shutdown() and not success and count<args.steps:
        if time.monotonic()-t0>args.max_wall_seconds:
            terminal_reason="wall_time_limit"
            break
        o=get_observation()
        if o is None:raise RuntimeError("stale observation")
        q=list(o["sensor"].joint_data.joint_q)
        left=list(o["hand"].left_hand_positions)
        right=list(o["hand"].right_hand_positions)
        if len(q)<18 or len(left)<11 or len(right)<11:
            raise RuntimeError("invalid 7/11 DOF state")
        left_q=[float(x) for x in q[4:11]]
        right_q=[float(x) for x in q[11:18]]
        state=right_q+[float(x)/1.57 for x in right[:11]]
        send_obj(conn,{"head":o["head"],"right":o["right"],"state":state})
        action=recv_obj(conn)
        if not isinstance(action,(list,tuple)) or len(action)!=21:
            raise RuntimeError("expected ACT 21D action")
        if not all(math.isfinite(float(x)) for x in action):
            raise RuntimeError("nonfinite policy action")
        if any(abs(float(x))>3.5 for x in action[:7]):
            raise RuntimeError("unsafe arm target")

        # Existing bridge publishes degrees; left arm held at current measured pose.
        msg=JointState()
        msg.header.stamp=rospy.Time.now()
        msg.name=["arm_joint_%d"%i for i in range(14)]
        msg.position=[math.degrees(float(x)) for x in left_q+list(action[:7])]
        arms.publish(msg)
        hand=SG100HandCommand()
        hand.header.stamp=rospy.Time.now()
        hand.control_mode=SG100HandCommand.MODE_JOINT_POSITION
        hand.left_hand_positions=left[:11]
        hand.right_hand_positions=[float(x)*1.57 for x in action[7:18]]
        hand.left_enable_mask=0
        hand.right_enable_mask=0x07FF
        hands.publish(hand)

        v=Twist()
        v.linear.x=max(-0.30,min(0.30,float(action[18])))
        v.linear.y=max(-0.20,min(0.20,float(action[19])))
        v.angular.z=max(-0.35,min(0.35,float(action[20])))
        base.publish(v)
        count+=1
        actual_hand=[float(x)/1.57 for x in right[:11]]
        target_hand=[float(x) for x in action[7:18]]
        arm_error=sum(abs(float(action[i])-right_q[i]) for i in range(7))/7.0
        hand_error=sum(abs(target_hand[i]-actual_hand[i]) for i in range(11))/11.0
        target_delta=(sum(abs(float(action[i])-previous_action[i]) for i in range(7))/7.0
                      if previous_action is not None else None)
        previous_action=[float(x) for x in action]
        with lock:
            xyz=base_pose
        entry={
            "step":count,
            "base_position_xyz_m":xyz,
            "elapsed_s":round(time.monotonic()-t0,3),
            "right_arm_measured_rad":right_q,
            "right_arm_target_rad":[float(x) for x in action[:7]],
            "right_sg100_measured_normalized":actual_hand,
            "right_sg100_target_normalized":target_hand,
            "base_velocity_raw":[float(x) for x in action[18:21]],
            "base_velocity_sent":[v.linear.x,v.linear.y,v.angular.z],
            "arm_target_tracking_mae_rad":round(arm_error,6),
            "hand_target_tracking_mae_normalized":round(hand_error,6),
            "arm_target_step_delta_rad":round(target_delta,6) if target_delta is not None else None
        }
        logs.write(json.dumps(entry,separators=(",",":"))+"\n")
        if count==1 or count%50==0:
            print("[ACTION] step={} elapsed={:.1f}s arm_track={:.4f} hand_track={:.4f} vx={:.3f} vy={:.3f} wz={:.3f}".format(
                count,entry["elapsed_s"],arm_error,hand_error,v.linear.x,v.linear.y,v.angular.z),flush=True)
        rate.sleep()
    if success:
        terminal_reason="success"
finally:
    base.publish(Twist())
    logs.close()
    conn.close()
    srv.close()
    print("[BRIDGE_DONE] steps={} wall_s={:.1f} success={} reason={} telemetry={}".format(
          count,time.monotonic()-t0,success,terminal_reason,args.telemetry),flush=True)
