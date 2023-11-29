Robotic Follow-Ahead with Obstacle and Occlusion Avoidance


Our methodology employs a high-level decision-making algorithm that strategically navigates a mobile robot in front of a person while 
avoiding collision and occlusion in any environment.

A DQN model (Q-model) of the robot is integrated with Monte Carlo Tree Search (MCTS) in order to enhance the performance of the decision-making process. 
The DQN package contains the code to train the Q-model. A simulation environment is written to input a state and action and calculate the corresponding next state and reward. Also, a pre-trained model is provided under the /models directory which is trained in an obstacle-free environment. The model inputs the relative pose of the robot with respect to the human and outputs the expected return of taking each action. The actions are going straight, turning left or right.

The MCTS package contains the codes to expand a tree at each time step and find the best navigational goal for the robot. The trained Q-model is used during the tree expansion process to evaluate each node of the tree.

Prerequisites:
1. An occupancy map of the environment
2. A module to output the position of the human. We used ZED2 camera and ZED-ros-wrapper package that publishes '/zed2/zed_node/obj_det/objects' topic. You can change the topic in the 'human_traj_pred.py' file.
3. ROS (Noetic version is preferred)

Getting Started:
1. Please refer to the installation.txt file to set up the proper version of the libraries.
2. Run the '.launch' file using roslaunch 


