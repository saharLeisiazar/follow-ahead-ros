from nodes import MCTSNode
import numpy as np
import copy


class MCTS:
    def __init__(self, node: MCTSNode):
        self.root = node
        self.params = node.params
        self.next_node_candidates = []
        self.candidate_ep3 = []

    def best_action(self, ep, traj):
        parent_node = self.root
        parent_node.update_ep(ep)
        current_ep = ep
        for i in range(self.params['num_expansion']):
            if (current_ep > ep):
                continue
            rollout_num = self.params["num_rollout"] - 1 - (ep - current_ep)
            if rollout_num:
                while not parent_node.is_fully_expanded():
                    child_node = parent_node.expand(ep)
                    if child_node == None:
                        continue
                    reward = child_node.rollout(ep, traj, rollout_num)
                    child_node.backpropagate(reward)
                    self.next_node_candidates.append(child_node)

            if i is not self.params['num_expansion']-1:
                if not len(self.next_node_candidates):
                    print("There is no best action/node to take")
                    return None, None
                next_node = self.best_candidate()
                parent_node = copy.deepcopy(next_node)
                parent_node.increase_ep(next_node.parent.ep)
                ep = copy.deepcopy(parent_node.ep)
                if ep > 2 : 
                    self.candidate_ep3.append(next_node)
                parent_node.state.state[1,:] = self.params['human_traj'][traj]['traj'][parent_node.ep]
                self.next_node_candidates.remove(next_node)

        # exploitation only
        return self.root.best_child(), self.best_candidate(c_param=0)


    def best_candidate(self, c_param= 1.4):  #c_param=1.4 good for exploration
        if c_param == 0:
            choices_weights = [
                (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(c.parent.n) / (c.n)))
                for c in self.candidate_ep3
            ]
            return self.candidate_ep3[np.argmax(choices_weights)]

        else:
            choices_weights = [
                (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(c.parent.n) / (c.n)))
                for c in self.next_node_candidates
            ]
            return self.next_node_candidates[np.argmax(choices_weights)]


    def all_children(self):
        return self.root.all_children()