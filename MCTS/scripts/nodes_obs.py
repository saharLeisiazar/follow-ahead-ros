import numpy as np
from collections import defaultdict
import util 


class MCTSNode(object):
    def __init__(self, state, params, parent=None, agent = None):
        self._number_of_visits = 0.
        self._number_of_episode = 0
        self._results = defaultdict(int)
        self.state = state
        self.parent = parent
        self.children = []
        self.children_reward = []
        self.params = params
        self.robot_act_list = np.copy(params["robot_actions"]).tolist()
        self.model = params['model']       
        self.agent = agent

    def select_action(self, state = None):
        if self.model != None:
            action_ind = np.argmax(util.to_numpy(self.model(util.from_numpy(state))))
        else:    
            if self.agent.is_model_saved and self.params['use_model']:
                action_ind = self.agent.forward(state)
            else:
                action_ind = np.random.choice(np.arange(len(self.params["robot_actions"])))  # will be replaced with policy network

        return  self.params["robot_actions"][action_ind], action_ind
        
    @property
    def q(self):
        return self.state.reward

    @property
    def n(self):
        return self._number_of_visits 

    @property
    def ep(self):
        return self._number_of_episode

    def expand(self, ep):
        action = self.robot_act_list.pop()
        next_state , reward , Rd, Ra = self.state.move(action[1], action[0])

        if  not util.is_occluded(self.state.state[0,:2], next_state.state[0,:2]):
            child_node = MCTSNode(next_state, self.params, parent=self, agent = self.agent)
            self.children.append(child_node)
            self.children_reward.append([reward, Rd, Ra])
        else:
            child_node = None

        return child_node


    def rollout(self, ep, traj, rollout_num, TS=0.4):
        current_rollout_state = np.copy(self.state.state)
        sum_of_rewards = 0

        for _ in range(rollout_num):
                current_rollout_state[1:] = self.params['human_traj'][traj]['traj'][ep+1]
                action, action_ind = self.select_action(state = current_rollout_state[1,:]-current_rollout_state[0,:])
                new_s, reward,_,_ = self.state.calculate_new_state(current_rollout_state, action[0]*TS, action[1]*TS,)
                sum_of_rewards += reward
                current_rollout_state = np.copy(new_s)
                ep +=1

        return sum_of_rewards / rollout_num

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self.state.reward += result

        if self.parent:
            self.parent.backpropagate(result)


    def is_fully_expanded(self):
        return not len(self.robot_act_list)

    def best_child(self, c_param=0):
        choices_weights = np.array([
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ])

        return [self.children[np.argmax(choices_weights)], self.children_reward[np.argmax(choices_weights)] , np.argmax(choices_weights)]

    def all_children(self):
        return [self.children[i].state.state for i in range(len(self.children))]
    
    
    def increase_ep(self, parent_ep):
        self._number_of_episode += 1 + parent_ep

    def update_ep(self, new_ep):
        self._number_of_episode = new_ep  