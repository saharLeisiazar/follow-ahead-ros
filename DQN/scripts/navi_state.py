import numpy as np
import math 
import util as ptu


class navState(object):

    def __init__(self, state, next_to_move=1):
        self.state = state 
        self.reward = 0
        self.next_to_move = next_to_move


    def nextToMove(self):
        return self.next_to_move

    def move(self, angle, dist):
        new_state, reward = self.calculate_new_state(self.state, angle, dist)
        next_to_move = -1 if self.next_to_move == 1 else 1
        return navState(new_state, next_to_move) , reward

    def calculate_new_state(self, state, angle, dist, next_to_move = 1 ):
        ind = 0 if next_to_move == 1 else 1  # zero for moving robot, one for moving human
        
        new_s = np.copy(state)
        new_s[ind,0] = state[ind,0] + dist * math.cos(angle + state[ind,2])
        new_s[ind,1] = state[ind,1] + dist * math.sin(angle + state[ind,2])
        new_s[ind,2] = angle + state[ind,2]
        reward = self.calculate_reward(new_s) if next_to_move == 1 else None

        return new_s, reward

    def calculate_reward(self, s):
        D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human
        beta = np.arctan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)           
        alpha = np.absolute(s[1,2] - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector         
   
        ######### distance reward
        # if (D <= 1 and D > 0.5):          
        #     Rd = -(1-D)
        # elif (D <= 2 and D > 1):          
        #     Rd = 0 #1*(0.5-np.absolute(D-1.5)) 
        # elif (D <= 5 and D > 2):          
        #     Rd = -0.25*(D-1)   
        # else:
        #     Rd = -1 
               
        ######## angle reward
        Ra = 1 * (45 - alpha)/45

        # ######## obstacle reward
        # if s.shape[0] == 3:
        #     do = np.linalg.norm([s[0,:2]- s[2, :2]])
        #     if do >0.4:
        #         Ro = 0
        #     elif do <=0.4 and do >0.2:
        #         Ro = 5* do - 2    
        #     else:
        #         Ro = -1     
        # else:
        #     Ro = 0    
        # beta = (s[0,2]- s[1, 2]) * 180 / np.pi
        # if beta < 20:
        #     Rb = 0.25*(20-beta)
        # else:
        #     Rb = -1    

        return min(max( Ra , -1) , 1)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
