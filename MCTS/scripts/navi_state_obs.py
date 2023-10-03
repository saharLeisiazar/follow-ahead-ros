import numpy as np
import math 
import util 


class navState(object):

    def __init__(self, state):
        self.state = state 
        self.reward = 0

    def move(self, Vx, W, TS = 0.5):
        dist = Vx * TS 
        angle = W * TS 
        new_state, reward , Rd, Ra = self.calculate_new_state(self.state, angle, dist)
        return navState(new_state) , reward, Rd, Ra

    def calculate_new_state(self, state, angle, dist):
        ind = 0   # zero for moving robot, one for moving human
        
        new_s = np.copy(state)
        new_s[ind,0] = state[ind,0] + dist * math.cos(angle + state[ind,2])
        new_s[ind,1] = state[ind,1] + dist * math.sin(angle + state[ind,2])
        new_s[ind,2] = angle + state[ind,2]
        reward , Rd, Ra = self.calculate_reward(new_s)

        return new_s, reward, Rd, Ra

    def calculate_reward(self, s):
        D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human
        beta = np.arctan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)           
        alpha = np.absolute(s[1,2] - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector         
   
        if (D <= 1 and D > 0.5):          
            Rd = -(1-D)
        elif (D <= 2 and D > 1):          
            Rd = 0
        elif (D <= 5 and D > 2):          
            Rd = -0.25*(D-1)   
        else:
            Rd = -1 

        Ra = 0.5* ((45 - alpha)/45)

        if util.is_occluded(s[0,:2], s[1, :2]):
            # print("found obstacle at:")
            # print(s)
            return -1, 0, -1

        return min(max(Ra + Rd , -1) , 1)  , Rd, Ra
    
