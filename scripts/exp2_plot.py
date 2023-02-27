import matplotlib.pyplot as plt
import numpy as np

file_add = "/home/sahar/Videos/good ones 2/"
file_name = "s3_filt.txt"
file = file_add + file_name


robot_x_list=[]
robot_y_list=[]
human_x_list=[]
human_y_list=[]
with open(file, 'r') as reader:
    for line in reader:
        robot_x_list.append(float(line.split()[0]))
        robot_y_list.append(float(line.split()[1]))
        human_x_list.append(float(line.split()[2]))
        human_y_list.append(float(line.split()[3]))

# print(robot_x_list) 
# print(robot_y_list) 
# print(human_x_list) 
# print(human_y_list) 

distance_list = []
alpha_list =[]

for i in range(len(robot_x_list)):
    D = np.linalg.norm([robot_x_list[i]-human_x_list[i] ,robot_y_list[i]-human_y_list[i]]) #distance_to_human
    distance_list.append(D)
    beta = np.arctan2(robot_y_list[i] - human_y_list[i] , robot_x_list[i] - human_x_list[i])   # atan2 (yr - yh  , xr - xh)   

    if i == 0:
        human_theta = np.arctan2(human_y_list[i+1] - human_y_list[i] , human_x_list[i+1] - human_x_list[i])
    else:
        human_theta = np.arctan2(human_y_list[i] - human_y_list[i-1] , human_x_list[i] - human_x_list[i-1])   

    alpha = (human_theta - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector   
    alpha_list.append(alpha)      


mean_d = np.mean(distance_list)
mean_alpha = np.mean(alpha_list) 
std_d = np.std(distance_list)
std_alpha = np.std(alpha_list)

print("mean_d", mean_d)
print("std_d", std_d)
print("mean_alpha", mean_alpha)
print("std_alpha", std_alpha)

########## plot
# fig, ax = plt.subplots()

# ax.plot(robot_x_list, robot_y_list, 'r')
# ax.plot(human_x_list, human_y_list, 'b')


fig, axs = plt.subplots()


idx = 0

human_x_list = np.array(human_x_list)
human_y_list = np.array(human_y_list)
robot_x_list = np.array(robot_x_list)
robot_y_list = np.array(robot_y_list)

xy= np.concatenate((human_x_list.reshape(human_x_list.shape[0],1), human_y_list.reshape(human_x_list.shape[0],1), robot_x_list.reshape(robot_x_list.shape[0],1), robot_y_list.reshape(robot_y_list.shape[0],1)), axis=1)

for start, stop in zip(xy[:-1], xy[1:]):
    x, y ,x2, y2= zip(start, stop)
    max=270
    step=int(max/(len(human_x_list)-1))
    if idx == 0:
        axs.plot(x, y, color=plt.get_cmap("jet")(idx), label='Human')
        axs.plot(x2, y2, color=plt.get_cmap("jet")(idx), label='Robot', marker='o', linestyle=' ')
        axs.arrow(x[0], y[0], (x[1]-x[0])*0.6, (y[1]-y[0])*0.6, head_width=0.15, head_length=0.15)
        axs.arrow(x2[0], y2[0], (x2[1]-x2[0])*0.6, (y2[1]-y2[0])*0.6, head_width=0.15, head_length=0.15)
    else:
        axs.plot(x, y, color=plt.get_cmap("jet")(idx))
        axs.plot(x2, y2, color=plt.get_cmap("jet")(idx), marker='o', linestyle=' ')
        # if x[0] == human_x_list[10]:
        #     axs.arrow(x[0], y[0], (x[1]-x[0])*0.2, (y[1]-y[0])*0.2, head_width=0.15, head_length=0.15,fc='green', ec='green')
        #     axs.arrow(x2[0], y2[0], (x2[1]-x2[0])*0.2, (y2[1]-y2[0])*0.2, head_width=0.15, head_length=0.15,fc='green', ec='green')

        if x[0] == human_x_list[-2]:
            axs.arrow(x[0], y[0], (x[1]-x[0])*0.6, (y[1]-y[0])*0.6, head_width=0.15, head_length=0.15,fc='red', ec='red')
            axs.arrow(x2[0], y2[0], (x2[1]-x2[0])*0.6, (y2[1]-y2[0])*0.6, head_width=0.15, head_length=0.15,fc='red', ec='red')            
    idx += step



axs.axis('equal')
fig.savefig(file_add+file_name[:-3]+'png', transparent=True)


fig, axs = plt.subplots(1)
cmaps = plt.get_cmap("jet")
pcm = axs.pcolormesh(np.random.random((1, 1)), cmap=cmaps, vmin = 0, vmax = len(human_x_list)/2)
fig.colorbar(pcm, ax=axs)

save_dir = file_add+ file_name[:-3] + '_exp3_bar.png'
fig.savefig(save_dir)

