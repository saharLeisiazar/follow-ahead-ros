from dqn_critic import DQNCritic
from argmax_policy import ArgMaxPolicy
import util as ptu
import numpy as np

class DQNAgent(object):
    def __init__(self, params):

        self.critic = DQNCritic(params)
        self.actor = ArgMaxPolicy(self.critic)
        self.target_update_freq = params['target_update_freq']
        self.num_param_updates = 0

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        if self.num_param_updates % self.target_update_freq == 0:
            self.critic.update_target_network()

        self.num_param_updates += 1

        return log        
    
    def forward(self, obs):
        return np.argmax(ptu.to_numpy(self.critic.q_net(ptu.from_numpy(self.critic.normilize(obs))))) 

    def save_net(self):
        self.critic.save_net()
    