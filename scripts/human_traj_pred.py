#!/usr/bin/env python

import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import Pose, PoseArray
import time


class human_traj_prediction():
    def __init__(self):
        rospy.init_node('prediction', anonymous=True)

        rospy.Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped, self.predictions_callback)
        self.pub_cur_pose = rospy.Publisher('/person_pose', Pose, queue_size = 1)
        self.pub_pred_pose_1 = rospy.Publisher('/person_pose_pred_1', Pose, queue_size = 1)
        self.pub_pred_pose_all = rospy.Publisher('/person_pose_pred_all', PoseArray, queue_size = 1)

        self.t_values = []
        self.x_values = []
        self.y_values = []
        self.init_time_1 = time.time()
        self.init_time_2 = time.time()
        self.init_time_3 = time.time()
        self.init_time_4 = time.time()
        self.init_time_5 = time.time()

    def predictions_callback(self,data):
        if len(data.objects) and data.objects[0].label == 'Person':
            position = data.objects[0].position

            poseArray = PoseArray()
            poseArray.header.frame_id = 'camera'
            poseArray.header.stamp = rospy.Time.now()

            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = position[2]
            self.pub_cur_pose.publish(pose)

            # sampling_freq = 10 #Hz
            # if time.time() > float(self.init_time_1) + (1/sampling_freq):  
            self.t_values.append(float(time.time()))
            self.x_values.append(position[0])
            self.y_values.append(position[1])

            if len(self.x_values) > 2*  15:     # sampling_freq:
                self.t_values.pop(0)
                self.x_values.pop(0)
                self.y_values.pop(0)

                # self.init_time = time.time()

            if len(self.x_values) > 20:
                coef_x = np.polyfit(self.t_values, self.x_values, 1)
                coef_y = np.polyfit(self.t_values, self.y_values, 1)
                p_x = np.poly1d(coef_x)
                p_y = np.poly1d(coef_y)

                for i in range(6):
                    msg = Pose()
                    msg.position.x = p_x(self.t_values[-1]+ 0.4*(i))
                    msg.position.y = p_y(self.t_values[-1]+ 0.4*(i))

                    poseArray.poses.append(msg)

                self.pub_pred_pose_all.publish(poseArray)
                self.pub_pred_pose_1.publish(msg)


if __name__ == '__main__':
    human_traj_prediction()
    rospy.spin()    