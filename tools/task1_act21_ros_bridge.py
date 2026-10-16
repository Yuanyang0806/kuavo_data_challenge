#!/usr/bin/env python3
"""Temporary 21D right-arm SG100+base adapter for official OpenLET model_entry."""
import argparse, math, pickle, socket, struct, threading, time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from kuavo_msgs.msg import sensorsData, SG100HandState, SG100HandCommand

p=argparse.ArgumentParser()
p.add_argument("--port",type=int,required=True)
p.add_argument("--bind",default="127.0.0.1")
p.add_argument("--steps",type=int,default=200)
args=p.parse_args()
rospy.init_node("openlet_task1_act21_bridge",anonymous=True)
lock=threading.Lock()
obs={k:None for k in ("head","right","sensor","hand")}
t_seen={k:0.0 for k in obs}
success=False

def update(key,obj):
    with lock:
        obs[key]=obj
        t_seen[key]=time.monotonic()
def success_cb(m):
    global success
    if m.data:success=True
rospy.Subscriber("/cam_h/color/image_raw/compressed",CompressedImage,lambda m:update("head",bytes(m.data)),queue_size=1)
rospy.Subscriber("/cam_r/color/image_raw/compressed",CompressedImage,lambda m:update("right",bytes(m.data)),queue_size=1)
rospy.Subscriber("/sensors_data_raw",sensorsData,lambda m:update("sensor",m),queue_size=1)
rospy.Subscriber("/sg100_hand_state",SG100HandState,lambda m:update("hand",m),queue_size=1)
rospy.Subscriber("/simulator/success",Bool,success_cb,queue_size=1)
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
try:
    rate=rospy.Rate(10)
    while not rospy.is_shutdown() and not success and count<args.steps:
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
        if count==1 or count%25==0:
            print("[ACTION] step={} vx={:.3f} vy={:.3f} wz={:.3f}".format(
                count,v.linear.x,v.linear.y,v.angular.z),flush=True)
        rate.sleep()
finally:
    base.publish(Twist())
    conn.close()
    srv.close()
    print("[BRIDGE_DONE] steps={} success={}".format(count,success),flush=True)
