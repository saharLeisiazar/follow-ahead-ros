from typing import Union
import torch
from torch import nn
import numpy as np

from replayBuffer import replay_buffer
import rospy
from nav_msgs.msg import OccupancyGrid

rospy.init_node('util', anonymous=False)
data = rospy.wait_for_message('/move_base/global_costmap/costmap', OccupancyGrid, timeout=None)
map_info= {}
map_info['height'] = data.info.height
map_info['width'] =  data.info.width
map_info['origin_x'] = data.info.origin.position.x
map_info['origin_y'] = data.info.origin.position.y
map_info['res'] = data.info.resolution

map_data = data.data

print(map_info)


def is_occluded(s1, s2):
    x1_ind = int(np.rint((s1[0] - map_info['origin_x']) / map_info['res']))
    y1_ind = int(np.rint((s1[1] - map_info['origin_y']) / map_info['res']))
    x2_ind = int(np.rint((s2[0] - map_info['origin_x']) / map_info['res']))
    y2_ind = int(np.rint((s2[1] - map_info['origin_y']) / map_info['res']))

    if x2_ind == x1_ind and y2_ind == y1_ind: return False

    slope = (y2_ind-y1_ind)/(x2_ind-x1_ind  + 1e-8) 
    obstacle_count =0

    range_step = np.sign((x2_ind-x1_ind))
    if range_step == 0: range_step = 1 

    x_list = range(x1_ind, x2_ind+1, range_step)
    for x in x_list:
        y0 = int(slope*(x-x1_ind) + y1_ind)
        for y in range(y0 -2, y0 +3):   
            if map_data[int(x + map_info['width'] * y)] > 0: 
                obstacle_count += 1

    return True if obstacle_count > 2 else False





BF = replay_buffer()
Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def add_sample_to_buffer(s, a, new_s, r, terminal = 0):
    return BF.add_sample(s, a, new_s, r, terminal)    

def sample_recent_data(Batch_size = 1):
    return BF.sample_recent_data(Batch_size)

def sample_random_data(Batch_size = 1):
    return BF.sample_random_data(Batch_size)


