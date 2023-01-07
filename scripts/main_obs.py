#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/sahar/catkin_ws/src/follow-ahead-ros/scripts')
from human_traj import trajectories

sys.path.insert(0, '/home/sahar/Follow-ahead-3/MCTS/scripts')
from nodes_obs import MCTSNode
from search_obs import MCTS
from navi_state_obs import navState
import util
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
import time
from visualization_msgs.msg import Marker


prediction_scale = 5  

#pose of camera in map frame
camera2map_trans = [0, 0, 0]  
camera2map_rot = [0,0, 0]

class node():
    def __init__(self):
        # rospy.init_node('follow', anonymous=False)
        # rospy.Subscriber("odom", Odometry, self.odom_callback, buff_size=1)
        rospy.Subscriber("/vicon/sahar_robot/sahar_robot", TransformStamped, self.odom_callback, buff_size=1)
        rospy.Subscriber("person_pose_pred_all", PoseArray, self.human_pose_callback, buff_size=1)

        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size =1)
        self.pub_human_pose = rospy.Publisher('/human_pose', Marker, queue_size = 1)

        self.robot_actions, _ = trajectories(15)
        self.state = np.zeros((2,3))
        self.MCTS_params = {}
        self.MCTS_params['robot_actions'] = self.robot_actions
        file_name = '[seven_actions][DQN]n_1000-nLayers_1-lsize_16-bs_512-lr_0.01-ep_15-test_1-tar_up100-aSize1-aStep30-2vel'
        model_directory = '/home/sahar/catkin_ws/src/follow-ahead-ros/model/' + file_name + '.pt'
        model = torch.load(model_directory)
        self.MCTS_params['model'] = model
        self.MCTS_params['use_model'] = True
        self.MCTS_params['num_expansion'] = 30
        self.MCTS_params['num_rollout'] = 5
        self.time = time.time()
        self.stay_bool = True
        print('hello')
        

    def odom_callback(self,data):
        robot_p = data.transform.translation
        robot_o = data.transform.rotation
        yaw = R.from_quat([0, 0, robot_o.z, robot_o.w]).as_euler('xyz', degrees=False)[2]
        self.state[0,:] = [robot_p.x, robot_p.y, yaw]

        # #################### comment this later
        # self.state = np.array([[1.5,0,0],[0.1,0,0]] ) 
        # Traj_list = np.array([[0.1,0,0],[0.2,0,0],[0.3,0,0],[0.4,0,0],[0.5,0,0],[0.6,0,0]])
        # Traj_dic={}
        # Traj_dic['traj'] = Traj_list
        
        # print("expanding the tree")
        # self.expand_tree([Traj_dic])


    def human_pose_callback(self, data):
        # self.state[0,:] = [0,0,0]  #comment this late
        human_traj = np.zeros((6,3))

        for i in range(len(data.poses)):
            human_traj[i,:] = [data.poses[i].position.x,
                                data.poses[i].position.y,
                                data.poses[i].orientation.z]

        rot_yaw = camera2map_rot[2]
        R_tran = [[np.cos(rot_yaw), -1*np.sin(rot_yaw), 0], [np.sin(rot_yaw), np.cos(rot_yaw), 0], [0, 0, 1]]

        Traj_list = np.zeros((6,3))
        for i in range(human_traj.shape[0]):
            new_elem= list(np.add(np.dot(R_tran , human_traj[i, :]), camera2map_trans[:2]+[rot_yaw]))
            if new_elem[2] > np.pi:  new_elem[2] -= 2*np.pi
            if new_elem[2] < -np.pi:  new_elem[2] += 2*np.pi

            Traj_list[i] = new_elem
        self.state[1,:] = Traj_list[0,:]

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.pose.position.x = Traj_list[0,0]
        m.pose.position.y = Traj_list[0,1]
        m.pose.orientation.z = R.from_euler('z', Traj_list[0,2], degrees=False).as_quat()[2]
        m.pose.orientation.w = R.from_euler('z', Traj_list[0,2], degrees=False).as_quat()[3]
        m.scale.x = 0.3
        m.scale.y = 0.1
        m.color.a = 1
        m.color.r = 255

        self.pub_human_pose.publish(m)

        Traj_dic={}
        Traj_dic['traj'] = Traj_list
        
        if time.time() > self.time: 
            self.expand_tree([Traj_dic])
            print()


    def expand_tree(self, human_traj):
        print()
        print("current state: ", self.state)
        if not self.stay():
            self.MCTS_params['human_traj'] = human_traj
            nav_state = navState(self.state) 
            node_human = MCTSNode(state=nav_state, params = self.MCTS_params, parent= None, agent = None)  
            mcts = MCTS(node_human)
            t1 = time.time()
            robot_best_node, leaf_node = mcts.best_action(0, 0)
            print("expansion time: ", time.time()-t1)
            print("leaf node: ", leaf_node.state.state)
            self.generate_goal(leaf_node.state.state)
            # if robot_best_node is not None:
            #     print('robot"action: ',self.robot_actions[robot_best_node[2]])
            #     self.generate_goal(robot_best_node[2])
            self.time += 1
        else:
            print("Waiting ...")

    def generate_goal(self, goal_state):
        # TS = 0.4
        # theta = self.robot_actions[idx][0] * TS
        # D = 3 if self.robot_actions[idx][1] == 0.25 else 3

        # goal_robot_frame = [D* np.cos(theta), D* np.sin(theta)]
        # print('goal_robot_frame', goal_robot_frame)

        # rot_yaw = self.state[0,2]
        # trans = self.state[0,:2]
        # R_tran = [[np.cos(rot_yaw), -1*np.sin(rot_yaw)], [np.sin(rot_yaw), np.cos(rot_yaw)]]

        # goal_map_frame= list(np.add(np.dot(R_tran , goal_robot_frame), trans))
        goal_map_frame = [goal_state[0,0], goal_state[0,1]]
        print('goal_map_frame', goal_map_frame)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = goal_map_frame[0]
        goal.pose.position.y = goal_map_frame[1]
        goal.pose.orientation.w = 1

        self.pub_goal.publish(goal)
  

    def stay(self):
        if self.stay_bool:
            s = self.state
            D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human
            beta = np.arctan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)           
            alpha = np.absolute(s[1,2] - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector         
    
            if D > 2:
                return True

            self.stay_bool = False
            return False


if __name__ == '__main__':
    node()
    rospy.spin()

