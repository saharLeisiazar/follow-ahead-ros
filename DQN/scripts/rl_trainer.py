import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from navi_state import navState
import pickle

class RL_Trainer(object):

    def __init__(self, params):
        self.params = params
        self.agent = self.params['agent_class']
        self.BF = self.params['buffer']
        self.test_reward_max = -10 #self.params['ep_length'] * self.params['max_reward_per_ep'] * -1
        self.eval_traj = []
        self.obst_pose = np.array([[2.5,0.3,0],[2.5,1,0],[2.5,-1,0]])



    def run_training_loop(self):
        all_logs = []
        test_mean = []
        test_std = []
        
        if self.params['use_dataset']:
            print("Loading dataset")
            with open(self.params['save_dir'] + "dataset/"+self.params['dataset_name'], "rb") as fp:   # Unpickling
                dataset = pickle.load(fp)
        
            print()
        for itr in range(self.params['n_iterations']):
            if itr % 10 == 0:
                print("----iteration: "+ str(itr))

            if self.params['use_dataset']:
                rand_indices = np.random.permutation(len(dataset))[:self.params['batch_size']]
                rollouts = [dataset[r] for r in rand_indices]
                ob_b, ac_b, next_ob_b, r_b, t_b = self.BF.convert_rollouts_to_list(rollouts)
            else:
                if itr % (self.params['n_iterations']/5) == 0:
                    self.BF.generate_sample(self.params['sample_gen_size'], itr) # times greater than batch size
                    ob_b, ac_b, next_ob_b, r_b, t_b = self.BF.sample_random_data(self.params['batch_size'])

            for _ in range(self.params['num_critic_updates_per_agent_update']):
                train_log = self.agent.train(ob_b, ac_b, r_b, next_ob_b, t_b)
                all_logs.append(train_log)
                
            mean, std , traj = self.test()
            test_mean.append(mean)
            test_std.append(std)
            
            if mean > self.test_reward_max: 
                self.agent.save_net()
                self.test_reward_max = mean
                self.eval_traj = traj
             
        if len(self.eval_traj): 
            self.plot_eval_traj()
            
        self.plotting(all_logs, test_mean, test_std)        
        return all_logs        

    def test(self):
        R = []
        all_traj = []
        
        for h in range(len(self.params['human_traj'])):
            traj = []
            ep_length = len(self.params['human_traj'][h])

            # if h == 0:
            #     init_state = np.concatenate([self.params['init_state'], self.obst_pose[0].reshape(1,3)], axis=0)
            # elif h == 1 or h == 4:
            #     init_state = np.concatenate([self.params['init_state'], self.obst_pose[1].reshape(1,3)], axis=0)
            # else:
            #     init_state = np.concatenate([self.params['init_state'], self.obst_pose[2].reshape(1,3)], axis=0)

            init_state = self.params['init_state']
            traj.append(init_state)

            state = np.copy(init_state)
            nav_state = navState(state)
            sum_of_rewards = 0
        
            for ep in range(ep_length):
                state[1,:] = self.params['human_traj'][h][ep]
                obs = state[1,:]-state[0,:]
                
                action = self.agent.forward(obs)
                acts = self.params["robot_actions"][action]
                new_state, reward = nav_state.calculate_new_state(state, acts[0], acts[1], next_to_move = 1)
                sum_of_rewards += reward
    
                traj.append(new_state)
                state = np.copy(new_state)

            all_traj.append(traj)
            R.append(sum_of_rewards)
            
        return np.mean(R), np.std(R), all_traj
    
            
            
    def plot_eval_traj(self): 
        
        # color = ['blue', 'deepskyblue', 'red', 'lightcoral', 'green', 'lightgreen']
        
        for h in range(len(self.eval_traj)):           
            robot_x = [xr[0][0] for xr in self.eval_traj[h]]
            robot_y = [yr[0][1] for yr in self.eval_traj[h]]
            human_x = [xh[1][0] for xh in self.eval_traj[h]]
            human_y = [yh[1][1] for yh in self.eval_traj[h]]
    
            # plt.figure()
            fig, ax = plt.subplots()
            obst_dim = 0.2
            if h == 0:
                rect = patches.Rectangle((self.obst_pose[0,0]- obst_dim/2, self.obst_pose[0,1]- obst_dim/2), obst_dim, obst_dim , linewidth=1,  facecolor='black')
            elif h == 1 or h == 4:
                rect = patches.Rectangle((self.obst_pose[1,0]- obst_dim/2, self.obst_pose[1,1]- obst_dim/2), obst_dim, obst_dim , linewidth=1,  facecolor='black')
            else:
                rect = patches.Rectangle((self.obst_pose[2,0]- obst_dim/2, self.obst_pose[2,1]- obst_dim/2), obst_dim, obst_dim , linewidth=1,  facecolor='black')


            cmap=plt.get_cmap("jet")
            for i in range(len(robot_x)):
                color = cmap(i/len(robot_x))
                if i != 0:
                    ax.plot(robot_x[i], robot_y[i], marker='o', c = color)
                    ax.plot(human_x[i], human_y[i], marker = 'x', c = color)
                    if i is not len(robot_x)-1:
                        ax.arrow(human_x[i], human_y[i], (human_x[i+1]-human_x[i])*0.3, (human_y[i+1]-human_y[i])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)
                        ax.arrow(robot_x[i], robot_y[i], (robot_x[i+1]-robot_x[i])*0.3, (robot_y[i+1]-robot_y[i])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)
                    else:
                        ax.arrow(human_x[i], human_y[i], (human_x[i]-human_x[i-1])*0.3, (human_y[i]-human_y[i-1])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)    
                        ax.arrow(robot_x[i], robot_y[i], (robot_x[i]-robot_x[i-1])*0.3, (robot_y[i]-robot_y[i-1])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)    

                else:
                    ax.plot(robot_x[i], robot_y[i], marker='o', c = color, label='Robot')
                    ax.plot(human_x[i], human_y[i], marker = 'x', c = color, label='Human')
                    ax.arrow(human_x[i], human_y[i], (human_x[i+1]-human_x[i])*0.3, (human_y[i+1]-human_y[i])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)
                    ax.arrow(robot_x[i], robot_y[i], (robot_x[i+1]-robot_x[i])*0.3, (robot_y[i+1]-robot_y[i])*0.3, head_width=0.05, head_length=0.05, fc=color, ec=color)
            
            # ax.add_patch(rect)
            ax.set(xlabel='X (meters)', ylabel='Y (meters)')
            ax.axis('equal')
            ax.legend()
            fig.savefig(self.params['save_dir']+ 'results/' + self.params['exp_name']+'-traj-'+ str(h) +'.png')

                            
    def plotting(self, logs, test_mean, test_std):
        
        train_loss = np.array([l['Training Loss']  for l in logs])
        iteration = np.arange(len(train_loss))+1
        plt.figure()
        plt.plot(iteration, train_loss)
        plt.xlabel('iterations')
        plt.ylabel('Training Loss')
        plt.savefig(self.params['save_dir']+ 'results/' + self.params['exp_name']+'-loss.png')
 
        iteration = np.arange(len(test_mean))+1
        plt.figure()
        plt.errorbar(iteration, test_mean, test_std, ecolor='lightgray')
        plt.xlabel('Iterations')
        plt.ylabel('Test reward')
        plt.savefig(self.params['save_dir']+ 'results/' + self.params['exp_name']+'-test_reward.png')  


                
                
                
                
                