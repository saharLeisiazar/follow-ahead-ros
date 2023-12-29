from torch import nn
import util as ptu
import torch
from torch import optim
from torch.nn import utils
from torch.optim.lr_scheduler import StepLR
import numpy as np

class DQNCritic():
    def __init__(self, params):
        self.params = params

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net = ptu.build_mlp(self.params['obs_dim'], self.params['ac_dim'], params['n_layer'], params['layer_size'])
        self.q_net_target = ptu.build_mlp(self.params['obs_dim'], self.params['ac_dim'], params['n_layer'], params['layer_size'])
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            self.params['learning_rate'],
        )
        self.scheduler = StepLR(self.optimizer, step_size=int(self.params['n_iterations'] * 0.75), gamma=0.1)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):

        ob_no = self.normilize(ob_no)
        next_ob_no = self.normilize(next_ob_no)
        
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        
        # TODO compute the Q-values from the target network 
        qa_tp1_values = self.q_net_target(next_ob_no)
        
        # if double_q:
        qa_t_next_values = self.q_net(next_ob_no)
            
        _, ind = qa_t_next_values.max(dim=1)
        q_tp1 = torch.gather(qa_tp1_values, 1, ind.unsqueeze(1)).squeeze(1)
        # else:
        #q_tp1, _ = qa_tp1_values.max(dim=1)

        target = reward_n + self.params['gamma'] * q_tp1 * (1-terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        #utils.clip_grad_value_(self.q_net.parameters(), 10)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)        
           
    def save_net(self):        
        torch.save(self.q_net, self.params['save_dir'] +'models/['+self.params['dataset_name']+']' +self.params['exp_name']+'.pt')
        print("Network saved")
        
    def normilize(self, data):           
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)  
        return (data - mean) / std

        
        
        
        
        
        
        
        
