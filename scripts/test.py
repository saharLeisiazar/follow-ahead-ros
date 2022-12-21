#!/usr/bin/env python3
from geometry_msgs.msg import PoseArray
import rospy



class node():
    def __init__(self):
        rospy.init_node('test', anonymous=False)

        rospy.Subscriber("human_traj_pred_all", PoseArray, self.callback)
        self.pub = rospy.Publisher('human_traj_pred_all_test', PoseArray, queue_size=1)

    def callback(self,data):
        p = PoseArray()
        p.header.frame_id = 'camera'
        p.header.stamp = data.header.stamp
        p.poses = data.poses

        self.pub.publish(p)
if __name__ == '__main__':
    node()
    rospy.spin()