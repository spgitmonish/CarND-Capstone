#!/usr/bin/env python

import math
import numpy as np
from eventlet import event
import rospy
import tf

from std_msgs.msg import Int32
from geometry_msgs.msg import Pose, Point, PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from __builtin__ import int

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''
TIME_PERIOD_PUBLISHED = 4. #sec
LOOKAHEAD_WPS = 10 # Number of waypoints we will publish. You can change this number
TIME_STEP = TIME_PERIOD_PUBLISHED / LOOKAHEAD_WPS
LIGHT_GREEN = 2

#finite states
FSM = { 'STOP':0,   # Car is stopped, waiting for light to clear
        'GO':1,     # Car is cruising, accelarate to reach speed limit
        'STOPPING':2} # Car is slowing down

class WaypointInfo(object):
    """
    Encapsulates waypoint data
    """
    def __init__(self, wp_obj, dist_to_next, frenet_s):
        self.wp = wp_obj
        self.width = dist_to_next
        self.s = frenet_s
        self.t = 0.

def distance2d(x1, y1, x2, y2):
    # returns euclidean distance
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

class WaypointUpdater(object):
    """
    Publishes waypoint data for navigating car:
    - slow down and stop at intersections
    - speed up and reach speed limit if there are no obstructions
    """
    def __init__(self):
        rospy.init_node('waypoint_updater', anonymous=True)

        # data received from other ROS nodes
        self.current_pose = None
        self.velocity = 0
        self.waypoints = None
        self.num_waypoints = 0

        # stores index of waypoint that is closest to car's current position
        self.nearestWaypointIndex = -1

        # contains waypoint of next RED traffic light, -1 if there is no light or if its green
        self.traffic_light = -1

        # initial state machine state
        self.fsm_state = FSM['GO']

        # read speed limit from ros parameter
        self.speed_limit = rospy.get_param('/waypoint_loader/velocity', 40) * 0.278 # KM/HR to M/S

        # subscribe to traffic light waypoint data from tl_detector
        rospy.Subscriber('/traffic_waypoint', Int32, self.stop_at_wp_cb)

        # subscribe to current position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        # subscribe to recieve waypoint data
        self.base_waypoint_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # subscribe to receive vehicle's velocity
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # Traffic light waypoint data from /vehicle/traffic_light
        # used only for debugging
        #self.tf_waypoints = None

        # publisher for sending out waypoints with velocity to twist_controller
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        rospy.spin()


    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.waypoints == None:
            return

        self.current_pose = msg

        # find nearest waypoint
        self.nearestWaypointIndex = self.getNearestWaypointIndex(self.current_pose)

        self.update_fsm()
        self.publish_waypoints()

    def update_fsm(self):
        """
        Updates state machine given current state of car and traffic light data
        """
        #rospy.loginfo("closest waypoint: %d; x,y: (%f,%f)", wp_start, *self.get_waypoint_coordinate(wp_start))
        # GO state - cruising normally
        if self.fsm_state == FSM['GO']:
            if self.traffic_light > 0:
                tf_distance = self.distanceToWaypoint(self.traffic_light)
                if tf_distance < self.get_braking_distance() and tf_distance > 0:
                    self.fsm_state = FSM['STOPPING']
                    rospy.loginfo("slowing for traffic light at waypoint: %d", self.traffic_light)

        # STOPPING state - slowing down at an intersection
        elif self.fsm_state == FSM['STOPPING']:
            if self.traffic_light < 0:
                self.fsm_state = FSM['GO']
            else:
                tf_distance = self.distanceToWaypoint(self.traffic_light)
                if self.velocity < 0.1:
                    self.fsm_state = FSM['STOP']
                    rospy.loginfo("distance: %f, velocity: %f", tf_distance, self.velocity)

        # STOP state: car has stopped, waiting for light to clear
        elif self.fsm_state == FSM['STOP']:
            if self.traffic_light < 0:
                self.fsm_state = FSM['GO']
                rospy.loginfo("going again")

    def publish_waypoints(self):
        """
        Creates a trjectory based on current state
        Broadcast waypoints to other ROS nodes on /waypoint_updater/final_waypoints
        """
        waypoints = []
        if self.fsm_state == FSM['GO']:
            waypoints = self.make_trajectory(self.speed_limit)
        elif self.fsm_state==FSM['STOPPING']:
            waypoints = self.make_trajectory(0)
        else:
            waypoints = self.stopped_waypoints()
            
        # return next n waypoints as a Lane pbject
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
        
    def make_trajectory(self,end_speed):
        """
        Select next n waypoints from current position and set target velocity
        """
        copyStart = self.nearestWaypointIndex
        next_wps = [] 
        for i in range(LOOKAHEAD_WPS):
            mapPoint = self.waypoints[(copyStart+i) % self.num_waypoints]
            wp = Waypoint()
            wp.pose.pose = mapPoint.wp.pose.pose
            wp.twist.twist.linear.x = end_speed
            next_wps.append(wp)
        return next_wps

    def stopped_waypoints(self):
        """
        Make an trjectory for the stopped car by repeating same waypoint n times
        and velocity set to 0
        """
        # waypoints for when the car is to stop -repeat same position with v=0
        self.output = []
        for t in np.arange(TIME_STEP, TIME_PERIOD_PUBLISHED, TIME_STEP ):
            p = Waypoint()
            p.pose.pose = self.current_pose.pose
            p.twist.twist.linear.x = 0.
            self.output.append(p)
        return self.output

    # update nearest waypoint index by searching nearby values
    # waypoints are sorted, so search can be optimized
    def getNearestWaypointIndex(self, pose): 
        # func to calculate cartesian distance
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        # previous nearest point not known, do exhaustive search
        # todo: improve with binary search
        if self.nearestWaypointIndex == -1:    
            r = [(dl(wp.wp.pose.pose.position, pose.pose.position), i) for i,wp in enumerate(self.waypoints)]
            return min(r, key=lambda x: x[0])[1]
        # previous nearest waypoint known, so scan points immediately after (& before)
        else:
            
            d = dl(self.waypoints[self.nearestWaypointIndex].wp.pose.pose.position, pose.pose.position)
            # scan right
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i + 1) % self.num_waypoints
                d2 = dl(self.waypoints[i].wp.pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                return i-1

            # scan left
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i - 1) % self.num_waypoints
                d2 = dl(self.waypoints[i].wp.pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                return i+1

            return self.nearestWaypointIndex# keep prev value

    # return distance from current position to specified waypoint
    def distanceToWaypoint(self, wp):
        if wp > self.nearestWaypointIndex:
            distances = [o.width for o in self.waypoints[self.nearestWaypointIndex:wp]]
        else:
            distances = [o.width for o in self.waypoints[self.nearestWaypointIndex:]] +  [o.width for o in self.waypoints[0:wp]]
        return sum(distances)

    def get_braking_distance(self):
        """
        Returns distance car will travel to come to a complete stop, given current speed.

        Ran trials and logged values for initial velocity (u), distance (s) and time (t) to come to a stop.
        Substituting values in s=ut+0.5at^2, we determined that decelaration is ~ 2.1m/s^2.
        t=-u/a for final velocity of 0.
        Therefore for any arbitrary u, s = -u^2 / 2*a.
        Add to that the distance from light (30m)         
        """
        return ((self.velocity**2) / (2*2.1)) + 35

    # Callback to recieve velocity messages
    def velocity_cb(self, vel):
        self.velocity = vel.twist.linear.x

    # Waypoint callback - data from /waypoint_loader
    # Since this is constant, so we cache it and un-subscribe after first call
    def waypoints_cb(self, base_lane):
        if self.waypoints == None:
            rospy.loginfo("waypoints_cb::%d", len(base_lane.waypoints))
            waypoints = []
            self.num_waypoints = len(base_lane.waypoints)

            d = 0.
            for i in range(0, self.num_waypoints):
                wp1_x = base_lane.waypoints[i].pose.pose.position.x
                wp1_y = base_lane.waypoints[i].pose.pose.position.y
                wp2_x = base_lane.waypoints[(i+1) % self.num_waypoints].pose.pose.position.x
                wp2_y = base_lane.waypoints[(i+1) % self.num_waypoints].pose.pose.position.y
                gap = distance2d(wp1_x, wp1_y, wp2_x, wp2_y)
                waypoints.append(WaypointInfo(base_lane.waypoints[i], gap, d))
                d += gap
            rospy.loginfo("track length: %f", d)
            self.waypoints = waypoints

            # unregister from waypoint callback - huge improvement in speed
            self.base_waypoint_sub.unregister()

    """
    Callback handler for /vehicle/traffic_light data
    - used only for testing and debugging
    def traffic_debug_cb(self, tf_arr):
        if self.tf_waypoints == None:
            tf_waypoints = []
            for tf in tf_arr.lights:
                tfp = tf.pose.pose.position
                closest_wp = min( ( distance2d(wp.wp.pose.pose.position.x, wp.wp.pose.pose.position.y, tfp.x, tfp.y), i) 
                    for i,wp in enumerate(self.waypoints) )[1]
                #closest_wp_pos = self.waypoints[closest_wp].wp.pose.pose.position
                #rospy.loginfo("TF %d @ %f,%f => %d, %f, %f", tf.state, tfp.x, tfp.y, closest_wp, closest_wp_pos.x, closest_wp_pos.y)
                tf_waypoints.append(closest_wp)
            self.tf_waypoints = tf_waypoints

        if self.nearestWaypointIndex != None and self.current_pose != None:
            # Get TF in front of current position
            R = [(tf_idx, tf_wp) for tf_idx, tf_wp in enumerate(self.tf_waypoints) if tf_wp > self.nearestWaypointIndex] + [(tf_idx, tf_wp) for tf_idx, tf_wp in enumerate(self.tf_waypoints) if tf_wp <= self.nearestWaypointIndex]
            tf_arr_idx = R[0][0]
            tf_wp = R[0][1]
            light_state = tf_arr.lights[tf_arr_idx].state

            distance_to_light = self.distanceToWaypoint(tf_wp)
            if distance_to_light < 100:
                if light_state == LIGHT_GREEN:
                    self.traffic_cb(-1)
                else:
                    self.traffic_cb(tf_wp)
            else:
                self.traffic_cb(-1)
    """

    # callback to receive traffic light waypoint from tl_detector
    def stop_at_wp_cb(self, stopIndexInt32):
        self.traffic_light = stopIndexInt32.data
        #rospy.loginfo("waypoint stop at called with %d", self.traffic_light)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # get velocity of waypoint object
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    # set velocity at specified waypoint index
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # Return coordinates for a given waypoint
    def get_waypoint_coordinate(self, wp):
        return (self.waypoints[wp].wp.pose.pose.position.x, self.waypoints[wp].wp.pose.pose.position.y)
    
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