#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.base_lane = None

        rospy.spin()

    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.base_lane.waypoints == None:
            return

        #rospy.loginfo("pose_cb::x:%f,y:%f,z:%f; qx:%f,qy:%f,qz:%f,qw:%f", 
        #    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        #    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        self.current_pose = msg

        # find nearest waypoint
        self.updateNearestWaypointIndex(self.base_lane.waypoints, self.current_pose)
        wp1 = self.nearestWaypointIndex
        rospy.loginfo("closest: %d", wp1)
        waypoints = self.base_lane.waypoints[wp1:(wp1 + LOOKAHEAD_WPS)%len(self.base_lane.waypoints)]
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)

        #1. Calculate Frenet coordinates for current_pose

        #2. Calculate velocity in Frenet space

        #3. FSM to plan route

        #4. Compute final Frenet coordinate at time Tr (end of trajectory)

        #5. Fit polynomical jerk minimizing trajectory

        #6. Select points for spline, convert them to map coordinates

        #7. Generate splines for X and Y

        #7. Generate map coordinate points as fixed time intervals (t=0.2)




    # update nearest waypoint index by searching nearby values
    def updateNearestWaypointIndex(self, waypoints, pose):  
        # func to calculate cartesian distance
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        # previous nearest point not known, do exhaustive search
        # todo: improve with binary search
        if self.nearestWaypointIndex == -1:    
            r = [(dl(wp.pose.pose.position, pose.pose.position), i) for i,wp in enumerate(waypoints)]
            self.nearestWaypointIndex = min(r, key=lambda x: x[0])[1]
            return

        # previous nearest waypoint known, so scan points immediately after (& before)
        else:
            numpoints = len(waypoints)
            d = dl(waypoints[self.nearestWaypointIndex].pose.pose.position, pose.pose.position)
            # scan right
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i + 1) % numpoints
                d2 = dl(waypoints[i].pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                self.nearestWaypointIndex = i-1
                return

            # scan left
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i - 1) % numpoints
                d2 = dl(waypoints[i].pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                self.nearestWaypointIndex = i+1
                return

            return # keep same


    # Waypoint callback - data from /waypoint_loader
    # I expect this to be constant, so we cache it and dont handle beyond 1st call
    def waypoints_cb(self, base_lane):
        if self.base_lane == None:
            rospy.loginfo("waypoints_cb::%d", len(base_lane.waypoints))
            self.nearestWaypointIndex = -1
            self.base_lane = base_lane

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # get velocity of waypoint object
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    # set velocity at specified waypoint index
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # arguments: wapoints and two waypoint indices
    # returns distance between the two waypoints
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
