#!/usr/bin/env python3
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray
import rospy
from visualization_msgs.msg import Marker, InteractiveMarkerPose,ImageMarker


class node():
    def __init__(self):
        rospy.init_node('test', anonymous=False)

        rospy.Subscriber("test_topic", String, self.callback, buff_size=1)

        self.pub_goal_vis = rospy.Publisher('/goal_vis', ImageMarker, queue_size =1)

    def callback(self,data):
        # g = Marker()
        # g.header.frame_id = 'map'
        # g.header.stamp = rospy.Time.now()
        # g.type = 0
        # g.pose.position.x = 3
        # g.pose.position.y = 0
        # g.scale.x = 0.3
        # g.scale.y = 0.3
        # g.color.a = 1
        # g.color.g = 0
        # g.color.b = 0.5
        # g.color.r = 1

        g = ImageMarker()
        g.header.frame_id = 'map'
        g.header.stamp = rospy.Time.now()
        g.position.x = 0
        g.position.y = 0
        g.position.z = 0
        g.scale=1




        print(g)
        self.pub_goal_vis.publish(g)

if __name__ == '__main__':
    node()
    rospy.spin()