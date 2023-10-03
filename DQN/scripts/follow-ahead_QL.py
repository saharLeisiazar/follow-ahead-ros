from rl_trainer import RL_Trainer
from dqn_agent import DQNAgent
from navi_state import navState
from replayBuffer import replay_buffer
import numpy as np

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', '-t', type=int, default=15)
    parser.add_argument('--batch_size', '-bs', type=int, default=218)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--target_update_freq', '-tuf', type=int, default=50)
    parser.add_argument('--action_size', type=int, default= 1)
    parser.add_argument('--action_step', type=float, default= 45)
    parser.add_argument('--n_layer', '-nl', type=int, default= 1)
    parser.add_argument('--layer_size', '-ls', type=int, default= 16)
    parser.add_argument('--learning_rate', '-lr', type=float, default= 1e-2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--n_iterations', '-n', type=int, default= 500)
    parser.add_argument('--ep_length', '-ep', type=int, default= 15)
    parser.add_argument('--move_forward_dist', type=int, default= 0.15)
    parser.add_argument('--test_size', type=int, default= 1)
    parser.add_argument('--generate_random_sample', type=int, default= 0)
    parser.add_argument('--dataset_size', type=int, default= 1000000)
    parser.add_argument('--use_dataset', type=int, default= 1)
    parser.add_argument('--dataset_name', type=str, default= 'angle_45(dt:0.5)[3actions]')
    args = parser.parse_args()
    params = vars(args)

    trainer = Q_Trainer(params)
    if not params['generate_random_sample']:
        trainer.run_training_loop()
    else:
        trainer.generate_random_samples()

class Q_Trainer(object):

    def __init__(self, params):
        self.params = params
        robot_angle_acts = np.arange(-self.params["action_size"], self.params["action_size"]+1) * self.params["action_step"] * np.pi /180
        robot_vel_acts = np.array([0.3])
        robot_actions = []
        for i in range(len(robot_vel_acts)):
            for j in range(len(robot_angle_acts)):
                robot_actions.append([robot_angle_acts[j], robot_vel_acts[i]])

        self.params["robot_actions"] = robot_actions
        self.params['ac_dim'] = (2* self.params['action_size'] + 1) * len(robot_vel_acts)
        self.params['obs_dim'] = 3
        human_trajectories = self.generate_human_traj()
        self.agent = DQNAgent(params)

        self.params['robot_vel_acts'] = robot_vel_acts
        self.params['sample_gen_size'] = 5
        self.params['agent_class'] = self.agent
        self.params['robot_angle_acts'] = robot_angle_acts
        self.params['human_traj'] = human_trajectories
        self.params['initial_state'] = np.array([[1.5,0,0],[0,0,0]])
        self.params['exp_name'] =  '[DQN]' + 'dataset:' + (params['dataset_name']) +'n:'+str(params['n_iterations'])+ '-nLayers:'+str(params['n_layer'])+ '-lsize:'+str(params['layer_size'])+ '-bs:'+str(params['batch_size'])+ '-lr:'+ str(params['learning_rate']) + '-ep:'+ str(params['ep_length'])+ '-test:'+ str(params['test']) + '-tar_up'+ str(params['target_update_freq']) + '-aSize' + str(params['action_size']) + '-aStep' + str(params['action_step'])
        self.params['save_dir'] = '/home/sahar/Follow-ahead-3/DQN/'
        self.params['init_state'] = np.array([[1.5, 0, 0],[0, 0, 0]])
        self.params['max_reward_per_ep'] = 0.5
        self.params['navstate'] = navState
        
        self.BF = replay_buffer(params= params)
        self.params['buffer'] = self.BF
        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop()


    def generate_random_samples(self):
        # sample_size = self.params['dataset_size']
        # human_traj = self.params['human_traj']
        # min_reward = 0.1
        # for t in range(len(human_traj)):
        #     for s in range(int(sample_size/len(human_traj))):
        #         if s % 1000 == 0:
        #             print('t is: ', t, "s is: " ,s)
                    
        #         state = np.copy(self.params['init_state'])
        #         nav_state = navState(state)

        #         for ep in range(len(human_traj[t])):
        #             state[1,:] = human_traj[t][ep]
        #             action = np.random.choice(self.params["ac_dim"])
        #             robot_move = self.params['robot_actions'][action]
        #             new_state, reward = nav_state.calculate_new_state(state, robot_move[0], robot_move[1] , next_to_move = 1)
        #             if reward > min_reward:    self.BF.add_sample(state, action, new_state, reward, 0)
        #             state = np.copy(new_state)
        min_reward = -1
        for s in range(self.params['dataset_size']):
            if s % 10000 == 0: print("s is: " ,s)
            robot_pose = [2,0,np.random.uniform(low=-60*np.pi/180, high=60*np.pi/180)]
            human_pose = [np.random.uniform(low=0, high=1), np.random.uniform(low=-1, high=1), np.random.uniform(low=-60*np.pi/180, high=60*np.pi/180)]
            # obst_pose = [np.random.uniform(low=2.5, high=3), np.random.uniform(low=-1, high=1), 0]

            state = np.array([robot_pose,human_pose])
            nav_state = navState(state)
            action = np.random.choice(self.params["ac_dim"])
            robot_move = self.params['robot_actions'][action]
            new_state, reward = nav_state.calculate_new_state(state, robot_move[0], robot_move[1] , next_to_move = 1)
            if reward > min_reward:    self.BF.add_sample(state, action, new_state, reward, 0)
        self.BF.save_to_file()

    def generate_human_traj(self):
        traj_length = int(self.params['ep_length'] * 1.5)
        dis = self.params['move_forward_dist']
        all_traj = []

        #########straight line##########
        human_traj = [[0,0,0]]
        for i in range(1, traj_length): human_traj.append([human_traj[i-1][0]+ dis*2, 0, 0])
        all_traj.append(human_traj[1:])
            
        ########circle##########    
        for i in range(2):
            k = 1 if i == 0 else -1
            human_traj = [[0,0,0]]
            alpha = self.params["action_step"] * np.pi /180 / 4. *k
            for i in range(1, traj_length): human_traj.append([human_traj[i-1][0] + dis * np.cos(alpha + human_traj[i-1][2]) , human_traj[i-1][1] + dis * np.sin(alpha + human_traj[i-1][2]), alpha + human_traj[i-1][2]])
            all_traj.append(human_traj[1:])

        ##########waveform#########
        # human_traj = [[0,0,0]]
        # for i in range(1, traj_length): 
        #     x = human_traj[i-1][0] + dis
        #     y = 0.5 * np.cos(x * np.pi / 2 + np.pi ) + 0.5
        #     human_traj.append([x, y, 0])
        #     human_traj[i-1][2] = np.arctan2(y - human_traj[i-1][1], 2 - human_traj[i-1][0]) 
        # human_traj[i][2] = human_traj[i-1][2]
        # all_traj.append(human_traj[1:])

        return all_traj


if __name__ == "__main__":
    main()
    print('Done!')