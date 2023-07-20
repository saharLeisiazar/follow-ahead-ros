#!/usr/bin/env python3

import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped , PoseArray, TransformStamped, Pose
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class human_traj_prediction():
    def __init__(self):
        rospy.init_node('prediction', anonymous=True)
        
        rospy.Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped, self.predictions_callback)
        self.pub_cur_pose = rospy.Publisher('/person_pose', PoseStamped, queue_size = 1)
        self.pub_pred_pose_all = rospy.Publisher('/person_pose_pred_all', PoseArray, queue_size = 1)

        self.t_values = []
        self.x_values = []
        self.y_values = []
        self.p_x_filt = []
        self.p_y_filt = []
        self.init_time = time.time()


    def predictions_callback(self,data):
        if len(data.objects) and data.objects[0].label == 'Person':
            position = data.objects[0].position

            self.p_x_filt.append(position[0])
            self.p_y_filt.append(position[1])

            # to filter the noise
            if len(self.p_x_filt) > 3:
                self.p_x_filt.pop(0)
                self.p_y_filt.pop(0)

            x = np.mean(self.p_x_filt)    
            y = np.mean(self.p_y_filt) 



            poseArray = PoseArray()
            poseArray.header.frame_id = 'camera'
            poseArray.header.stamp = rospy.Time.now()

            #publishing currect and filtered pose of the person
            pose = PoseStamped()
            pose.header.frame_id = 'camera'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            self.pub_cur_pose.publish(pose)


            self.t_values.append(float(time.time()))
            self.x_values.append(x)
            self.y_values.append(y)

            if len(self.x_values) > 4*  15:     # sampling_freq:
                self.t_values.pop(0)
                self.x_values.pop(0)
                self.y_values.pop(0)

            # Fitting a polynomial to predict the future motion of the person 
            if len(self.x_values) > 30:
                coef_x = np.polyfit(self.t_values, self.x_values, 10)
                coef_y = np.polyfit(self.t_values, self.y_values, 10)
                p_x = np.poly1d(coef_x)
                p_y = np.poly1d(coef_y)

                # Extrapolation
                for i in range(7):
                    dt=0.5
                    msg = Pose()
                    msg.position.x = p_x(self.t_values[-1]+ dt*(i))
                    msg.position.y = p_y(self.t_values[-1]+ dt*(i))

                    theta_list=[]
                    for j in range(1,5):
                        theta_list.append(np.arctan2(msg.position.y- p_y(self.t_values[-1]+ dt*(i-j)) , msg.position.x -p_x(self.t_values[-1]+ dt*(i-j))))
                    msg.orientation.z = np.mean(theta_list)
                    poseArray.poses.append(msg)

                self.pub_pred_pose_all.publish(poseArray)



if __name__ == '__main__':
    human_traj_prediction()
    rospy.spin()    
