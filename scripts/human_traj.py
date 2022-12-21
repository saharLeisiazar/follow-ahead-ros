import numpy as np
from numpy import linalg as LA

def trajectories(length):
    straight_line = True
    circle = True
    waveform = True

    human_traj_list = []
    vel_reduction = 1

    ###straight line
    if straight_line:
        line = [[0,0,0]]
        line_lin_vel = []
        line_ang_vel = []

        for i in range(1, length+10): 
            line.append([line[i-1][0]+ 0.1 *2, 0, 0])

            dx, _ , _= np.subtract(line[i],line[i-1])
            vel = dx / 0.4
            line_lin_vel.append(vel/vel_reduction)
            line_ang_vel.append(0)

        traj={}
        del line[0]
        traj['traj'] = line
        traj['vel'] = line_lin_vel
        traj['ang'] = line_ang_vel
        human_traj_list.append(traj)

    ###circle
    if circle:
        circle = [[0,0,0]]
        circle_lin_vel = []
        circle_ang_vel = []
        dis = 0.1
        alpha = 7.5 / 180 * np.pi

        for i in range(1, length+10): 
            x = circle[i-1][0] + dis * np.cos(alpha + circle[i-1][2])
            y = circle[i-1][1] + dis * np.sin(alpha + circle[i-1][2])
            theta = alpha + circle[i-1][2]
            circle.append([x, y, theta])

            dx, dy ,dtheta= np.subtract(circle[i],circle[i-1])
            vel = LA.norm([dx, dy]) / 0.4
            circle_lin_vel.append(vel/vel_reduction)

            if i > 1:
                ang = (circle[i-1][2] - circle[i-2][2])/0.4
                circle_ang_vel.append(ang)

        circle[i][2] = circle[i-1][2]
        circle_ang_vel.append(0)

        traj={}
        del circle[0]
        traj['traj'] = circle
        traj['vel'] = circle_lin_vel
        traj['ang'] = circle_ang_vel
        human_traj_list.append(traj)

    ###waveform
    if waveform:
        wave = [[0,0,0]]
        wave_lin_vel = []
        wave_angular_vel = []

        for i in range(1, length *2 +10): 
            x = wave[i-1][0] + 0.1
            y = 0.5 * np.cos(x * np.pi / 2 + np.pi ) + 0.5
            wave.append([x, y, 0])
            wave[i-1][2] = np.arctan2(y - wave[i-1][1], x - wave[i-1][0])  #prev_theta 

            dx, dy ,dtheta= np.subtract(wave[i],wave[i-1])
            vel = LA.norm([dx, dy]) / 0.4
            wave_lin_vel.append(vel/vel_reduction)

            if i > 1:
                ang = (wave[i-1][2] - wave[i-2][2])/0.4
                wave_angular_vel.append(ang)

        wave[i][2] = wave[i-1][2]
        wave_angular_vel.append(0)

        traj={}
        del wave[0]
        traj['traj'] = wave
        traj['vel'] = wave_lin_vel
        traj['ang'] = wave_angular_vel
        human_traj_list.append(traj)



    ############ robot action space with linear velocity in x and angular vel
    TS = 0.4
    robot_angle_acts = np.arange(-1, 2) * 30 * np.pi /180 /TS
    robot_vel_acts = np.array([0.1, 0.2]) /TS
    robot_actions = [[0,0]]
    for i in range(len(robot_vel_acts)):
        for j in range(len(robot_angle_acts)):
            robot_actions.append([robot_angle_acts[j], robot_vel_acts[i]])

    ############ robot action space with linear velocity in x and y
    # TS = 0.4
    # scale = 1 #to adjust with pybullet speed
    # robot_vel_x = np.array([-0.2, -0.1, 0, 0.1, 0.2]) /TS * scale
    # robot_vel_y = np.array([-0.2, -0.1, 0, 0.1, 0.2]) /TS * scale 
    # # robot_vel_x = np.array([0.1, 0.2]) /TS * scale
    # # robot_vel_y = np.array([0]) /TS * scale 
    # robot_actions = []
    # for i in range(len(robot_vel_y)):
    #     for j in range(len(robot_vel_x)):
    #         robot_actions.append([robot_vel_x[j], robot_vel_y[i], 0])


    return robot_actions, human_traj_list      

if __name__ == '__main__':
    robot_actions, human_traj_list   = trajectories(15)
    print(robot_actions)
