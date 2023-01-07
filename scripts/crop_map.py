#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid




class node():
    def __init__(self):
        rospy.init_node('map_crop', anonymous=False)
        rospy.Subscriber("map", OccupancyGrid, self.callback, buff_size=1)
        self.pub = rospy.Publisher("/croped_map", OccupancyGrid, queue_size=1)


    def callback(self, map):
        print("publishing")
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = map.header.frame_id
        msg.info.resolution = map.info.resolution
        
        new_data = []
        x_idx_min = 850
        x_idx_max = 1150
        y_idx_min = 900
        y_idx_max = 1150    

        msg.info.width = x_idx_max - x_idx_min  
        msg.info.height = y_idx_max - y_idx_min   

        msg.info.origin.position.x = map.info.origin.position.x + x_idx_min * map.info.resolution
        msg.info.origin.position.y = map.info.origin.position.y + y_idx_min * map.info.resolution
        msg.info.origin.orientation.w = 1

        for j in range(y_idx_min, y_idx_max):
            for i in range(x_idx_min, x_idx_max):
                new_data.append(map.data[int(i + map.info.width * j)])
       

        msg.data = new_data

        self.pub.publish(msg)
        




if __name__ == '__main__':
    node()
    rospy.spin()        