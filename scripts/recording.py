#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray,PoseStamped
import time
from geometry_msgs.msg import TransformStamped


import os
file_name = '/home/sahar/catkin_ws/src/follow-ahead-ros/scripts/record.txt'
os.remove(file_name)

class traj():
    def __init__(self):
        rospy.init_node('traj', anonymous=True)

        rospy.Subscriber("/vicon/sahar_robot/sahar_robot", TransformStamped, self.odom_callback, buff_size=1)
        rospy.Subscriber("/vicon/sahar_helmet/sahar_helmet", TransformStamped, self.human_pose_callback, buff_size=1)

        self.robot_x = 0
        self.robot_y = 0
        self.human_x = 0
        self.human_y = 0
        self.t1 = time.time()


    def odom_callback(self, data):
        self.robot_x = data.transform.translation.x
        self.robot_y = data.transform.translation.y
        # self.robot_x = data.pose.pose.position.x
        # self.robot_y = data.pose.pose.position.y

    def human_pose_callback(self, data):
        # self.human_x = data.pose.position.x
        # self.human_y = data.pose.position.y
        self.human_x = data.transform.translation.x
        self.human_y = data.transform.translation.y
        if time.time() - self.t1 > 0.5:
            self.t1 = time.time()
            f.write(str(self.robot_x))
            f.write('\t')
            f.write(str(self.robot_y))
            f.write('\t')
            f.write(str(self.human_x))
            f.write('\t')
            f.write(str(self.human_y))
            f.write('\n')
            self.t1 = time.time()




if __name__ == '__main__':
    traj()
    with open(file_name, 'a') as f:
        rospy.spin()  


    # robot_x_list=[]
    # robot_y_list=[]
    # human_x_list=[]
    # human_y_list=[]
    # with open('/home/sahar/catkin_ws/src/follow-ahead-ros/scripts/record.txt', 'r') as reader:
    #     for line in reader:
    #         robot_x_list.append(float(line.split()[0]))
    #         robot_y_list.append(float(line.split()[1]))
    #         human_x_list.append(float(line.split()[2]))
    #         human_y_list.append(float(line.split()[3]))
 
    # print(robot_x_list) 
    # print(robot_y_list) 
    # print(human_x_list) 
    # print(human_y_list) 