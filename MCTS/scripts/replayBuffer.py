import numpy as np
import pickle


class replay_buffer:
    def __init__(self, params= None):
        self.params = params
        self.rollout_list = []

        if params != None:
            self.agent = params['agent_class']
            self.nav_state = params['navstate']

    def add_sample(self, s, a, new_s, r, terminal = 0):

        rollout = {"observation": s[1,:]-s[0,:],
                    "action": a,
                    "next_observation": new_s[1,:]-new_s[0,:],
                    "reward": r,
                    "terminal": terminal}

        self.rollout_list.append(rollout)


    def buffer_size(self):
        return len(self.rollout_list)

    def sample_recent_data(self, Batch_size = 1):
        if len(self.rollout_list) > 10* Batch_size :
            self.rollout_list = self.rollout_list[-10*Batch_size:]

        s = min(Batch_size, self.buffer_size())
        rollouts = self.rollout_list[-s:]
        return self.convert_rollouts_to_list(rollouts) #ob_b , ac_b, next_ob_b, r_b, t_b

    def sample_random_data(self, Batch_size = 1):
        s = self.params['sample_gen_size'] +1
        if len(self.rollout_list) > s * Batch_size :
            self.rollout_list = self.rollout_list[-s * Batch_size:]

        rand_indices = np.random.permutation(len(self.rollout_list))[:Batch_size]
        rollouts = [self.rollout_list[r] for r in rand_indices]
        return self.convert_rollouts_to_list(rollouts) #ob_b , ac_b, next_ob_b, r_b, t_b
            
    def convert_rollouts_to_list(self, rollouts):
        ob_b = np.concatenate([rollout["observation"] for rollout in rollouts]).reshape(len(rollouts), 3)
        action_b = np.array([rollout["action"] for rollout in rollouts]) 
        next_ob_b = np.concatenate([rollout["next_observation"] for rollout in rollouts]).reshape(len(rollouts), 3)
        r_b = np.array([rollout["reward"] for rollout in rollouts]) 
        t_b = np.array([rollout["terminal"] for rollout in rollouts]) 
        
        return ob_b, action_b, next_ob_b, r_b, t_b    
    
    def save_to_file(self):
        with open("/home/sahar/Follow-ahead/dataset/"+ self.params['dataset_name'], "wb") as f:   #Pickling
            pickle.dump(self.rollout_list, f)
        
        
    def generate_sample(self, times, itr):
        print('Generating samples')
        for i in range(self.params['batch_size'] * times):
            if i % self.params['batch_size'] == 0:
                print(i)
    
            for h in range(len(self.params['human_actions'])):
                state = self.params['init_state']
                nav_state = self.nav_state(state)
                human_move = self.params['human_actions'][h]
                
                for ep in range(self.params['ep_length']):
                    state, _ = nav_state.calculate_new_state(state, human_move, self.params['robot_vel_acts'][0], next_to_move = -1)
                        
                    if itr < self.params['n_iterations'] * 0.7:
                        robot_move = np.random.choice(self.params['ac_dim'])
                    else:    
                        robot_move = self.agent.forward(state[1] - state[0])
                        
                    vel_ind = 0
                    while robot_move >= len(self.params['robot_angle_acts']):
                        robot_move -= len(self.params['robot_angle_acts'])
                        vel_ind +=0
                        
                    new_state, reward = nav_state.calculate_new_state(state, robot_move, self.params['robot_vel_acts'][vel_ind] , next_to_move = 1)
                    self.add_sample(state, robot_move, vel_ind, new_state, reward, 0)

        
        
        
        
        
        
        