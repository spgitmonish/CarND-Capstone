#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import Pose, Point, PoseStamped, TwistStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from eventlet import event

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''
TIME_PERIOD_PUBLISHED = 1. #sec
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
SPEED_LIMIT = 20.0 # m/s
TIME_TO_MAX = 5.0 # 0 to 50 in 20 sec
LIGHT_BREAKING_DISTANCE_METERS = 80 # meters

#FSM_GO = 0
#FSM_STOPPING = 1

#finite states
FSM = {'STOP':0,
        'GO':1,
        'STOPPING':2}

def clip(val, min, max):
    if val < min:
        return min
    if val > max:
        return max
    return val

def cartesian_distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

def theta_slope(w1, w2):
    x1 = w1.pose.pose.position.x
    y1 = w1.pose.pose.position.y
    x2 = w2.pose.pose.position.x
    y2 = w2.pose.pose.position.y
    return math.atan2((y2-y1), (x2-x1))

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoint_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.velocity = 0
        self.base_lane = None
        self.base_waypoints = None
        self.num_waypoints = 0
        self.base_waypoint_distances = None

        """
        light 1: closest waypoint: 289; x,y: (1145.720000,1184.640000)
        light 2: closest waypoint: 750; x,y: (1556.820000,1158.860000)

        """
        self.traffic_light = 750

        # initial state machine state
        self.fsm_state = FSM['GO']

        rospy.spin()

    def traffic_lights_off(self, timer_event):
        rospy.loginfo("traffic_light_cleared")
        self.traffic_light = None

    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.base_waypoints == None:
            return

        self.current_pose = msg

                # find nearest waypoint
        wp_start = self.getNearestWaypointIndex(self.current_pose)
        self.nearestWaypointIndex = wp_start

        distance = -1 
        if self.traffic_light != None:
            distance = sum(self.base_waypoint_distances[wp_start : self.traffic_light])
        self.fsm_state = self.update_fsm(self.fsm_state,distance)
        
        self.update_waypoints()

    
    def update_fsm(self, stateFSM, tf_distance):  
        #rospy.loginfo("closest waypoint: %d; x,y: (%f,%f)", wp_start, *self.get_waypoint_coordinate(wp_start))

        if stateFSM == FSM['GO']:
            if tf_distance < LIGHT_BREAKING_DISTANCE_METERS and tf_distance > 0:
                stateFSM = FSM['STOPPING']
                rospy.loginfo("slowing for traffic light at waypoint: %d", self.traffic_light)
                
        elif stateFSM == FSM['STOPPING']:        
            if tf_distance < 5 and tf_distance > -5:
                stateFSM = FSM['STOP']
                rospy.loginfo("stopping for traffic light at waypoint: %d", self.traffic_light)
                
        elif stateFSM == FSM['STOP']:  
            if self.traffic_light == None or tf_distance > 10:
                stateFSM = FSM['GO']
                rospy.loginfo("going again")
        
        return stateFSM
    
    def update_waypoints(self):
        waypoints = []
        if self.fsm_state == FSM['GO']:
            waypoints = self.accelerate()
        else:
            waypoints = self.decelerate(self.traffic_light)
            
        # return next n waypoints as a Lane pbject
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
    
    def accelerate(self):
        # accelaration
        #rospy.loginfo("accelarating: %f", a)
        a = clip((SPEED_LIMIT - self.velocity) / TIME_PERIOD_PUBLISHED, -SPEED_LIMIT/TIME_TO_MAX, SPEED_LIMIT/TIME_TO_MAX)
        return self.interpolate_waypoints(a)

    def decelerate(self, wp_end):
        """
        solve for -ve accelaration requried to stop car at given waypoint
            v = u + at
            subst v = 0, gives
            eq1. a = -u/t
            substitute in s=ut+0.5at^2
            s = ut/2
            t = 2s/u
            substitute back in eq1. gives a = -u^2/2s
        """
        if wp_end > self.nearestWaypointIndex:
            distances = self.base_waypoint_distances[self.nearestWaypointIndex:wp_end]
        else:
            distances = self.base_waypoint_distances[self.nearestWaypointIndex:] + self.base_waypoint_distances[0:wp_end]
        u = self.velocity
        s = sum(distances)
        a = -u**2 / (2*s)
        rospy.loginfo("decelarate: %f", a)

        return self.interpolate_waypoints(a)

    def interpolate_waypoints(self, a):
        output = []
        v = self.velocity

        x = []
        y = []
        z = []
        qw = []
        qx = []
        qy = []
        qz = []
        t = []
        wp_idx = self.nearestWaypointIndex
        max_s = (self.velocity * TIME_PERIOD_PUBLISHED) + (0.5 * a * TIME_PERIOD_PUBLISHED * TIME_PERIOD_PUBLISHED)
        s = 0
        while s < max_s:
            wp = self.base_waypoints[wp_idx]
            x.append(wp.pose.pose.position.x)
            y.append(wp.pose.pose.position.y)
            z.append(wp.pose.pose.position.z)
            qw.append(wp.pose.pose.orientation.w)
            qx.append(wp.pose.pose.orientation.x)
            qy.append(wp.pose.pose.orientation.y)
            qz.append(wp.pose.pose.orientation.z)
            if a == 0:
                t.append( s / self.velocity )
            else:
                t.append( (math.sqrt(self.velocity**2 + 2*a*s) - self.velocity) / a )
            s += self.base_waypoint_distances[wp_idx % self.num_waypoints]
            wp_idx += 1
        
        if len(x) < 2: # can't use splines, interpolate to next point
            wp = self.nearestWaypointIndex
            theta = theta_slope(self.base_waypoints[wp % self.num_waypoints], self.base_waypoints[(wp+1) % self.num_waypoints])

            for t in np.arange(0, LOOKAHEAD_WPS*0.02, 0.02):
                r = v*t + 0.5*a*(t**2)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                p = Waypoint()
                p.pose.pose.position.x = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.x + x)
                p.pose.pose.position.y = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.y + y)
                p.pose.pose.position.z = self.base_waypoints[wp % self.num_waypoints].pose.pose.position.z
                p.pose.pose.orientation = self.base_waypoints[wp % self.num_waypoints].pose.pose.orientation
                p.twist.twist.linear.x = float(clip(v+a*t,0,SPEED_LIMIT))
                output.append(p)
            return output

        #x_spline = interp1d(t, x, kind='cubic')
        #y_spline = interp1d(t, y, kind='cubic')
        k = 5 # qunitic
        if len(x) < 6:
            k = 3 # cubic
        if len(x) < 4:
            k = 1 # linear
        x_spline = splrep(t, x, k = k) 
        y_spline = splrep(t, y, k = k)
        z_spline = splrep(t, z, k = k)
        qw_spline = splrep(t, qw, k = k)
        qx_spline = splrep(t, qx, k = k)
        qy_spline = splrep(t, qy, k = k)
        qz_spline = splrep(t, qz, k = k)

        for t in np.arange(0, TIME_PERIOD_PUBLISHED, TIME_PERIOD_PUBLISHED / LOOKAHEAD_WPS ):
            p = Waypoint()
            p.pose.pose.position.x = float(splev(t, x_spline))
            p.pose.pose.position.y = float(splev(t, y_spline))
            p.pose.pose.position.z = float(splev(t, z_spline))
            p.pose.pose.orientation.w = float(splev(t, qw_spline))
            p.pose.pose.orientation.x = float(splev(t, qx_spline))
            p.pose.pose.orientation.y = float(splev(t, qy_spline))
            p.pose.pose.orientation.z = float(splev(t, qz_spline))
            p.twist.twist.linear.x = float(clip(v+a*t,0,SPEED_LIMIT))
            output.append(p)
        return output

    # update nearest waypoint index by searching nearby values
    # waypoints are sorted, so search can be optimized
    def getNearestWaypointIndex(self, pose):  
        # func to calculate cartesian distance
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        # previous nearest point not known, do exhaustive search
        # todo: improve with binary search
        if self.nearestWaypointIndex == -1:    
            r = [(dl(wp.pose.pose.position, pose.pose.position), i) for i,wp in enumerate(self.base_waypoints)]
            return min(r, key=lambda x: x[0])[1]
        # previous nearest waypoint known, so scan points immediately after (& before)
        else:
            
            d = dl(self.base_waypoints[self.nearestWaypointIndex].pose.pose.position, pose.pose.position)
            # scan right
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i + 1) % self.num_waypoints
                d2 = dl(self.base_waypoints[i].pose.pose.position, pose.pose.position)
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
                d2 = dl(self.base_waypoints[i].pose.pose.position, pose.pose.position)
                if d2 > d1: break
                d1 = d2
                found = True
            if found:
                return i+1

            return self.nearestWaypointIndex# keep prev value

    def velocity_cb(self, vel):
        self.velocity = vel.twist.linear.x
        #rospy.loginfo("velocity: %f", vel.twist.linear.x)

    # Waypoint callback - data from /waypoint_loader
    # I expect this to be constant, so we cache it and dont handle beyond 1st call
    def waypoints_cb(self, base_lane):
        if self.base_lane == None:
            rospy.loginfo("waypoints_cb::%d", len(base_lane.waypoints))
            self.nearestWaypointIndex = -1
            self.base_lane = base_lane
            self.base_waypoints = base_lane.waypoints
            self.num_waypoints = len(self.base_waypoints)
            self.base_waypoint_distances = []
            d = 0.
            pos1 = self.base_waypoints[0].pose.pose.position
            for i in range(1, self.num_waypoints + 1):
                pos2 = self.base_waypoints[i % self.num_waypoints].pose.pose.position
                gap = cartesian_distance(pos1,pos2)
                self.base_waypoint_distances.append(gap)
                pos1 = pos2
            rospy.loginfo("track length: %f", d)
            self.base_waypoint_sub.unregister()

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

    def get_waypoint_coordinate(self, wp):
        return (self.base_waypoints[wp].pose.pose.position.x, self.base_waypoints[wp].pose.pose.position.y)

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
