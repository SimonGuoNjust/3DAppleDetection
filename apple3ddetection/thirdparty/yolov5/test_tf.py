import numpy as np
import pyrealsense2 as rs
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf

def camera_tf_world(pub,ts_now):
    ts = TransformStamped()
    ts.header.frame_id = "camera"
    ts.header.stamp = rospy.Time.now()
    ts.child_frame_id = f"world"
    ts.transform.translation.x = 0
    ts.transform.translation.y = 0
    ts.transform.translation.z = -3
    qtn = tf.transformations.quaternion_from_euler(ts_now[0]/360*np.pi, ts_now[1]/360*np.pi, ts_now[2]/360*np.pi)
    ts.transform.rotation.x = qtn[0]
    ts.transform.rotation.y = qtn[1]
    ts.transform.rotation.z = qtn[2]
    ts.transform.rotation.w = qtn[3]
    pub.sendTransform(ts)

if __name__ == "__main__":
    rospy.init_node('apple_detector')
    pub = tf2_ros.TransformBroadcaster()
    ts_now = [0,0,0]
    while True:
        axis = input("aixs:")
        if axis == '-1':
            break
        value = input("value")
        ts_now[int(axis)] = int(value)
        camera_tf_world(pub,ts_now)


