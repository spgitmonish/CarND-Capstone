#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
from scipy import interpolate

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
SPEED_LIMIT = 20.0 # m/s
TIME_TO_MAX = 5.0 # 0 to 50 in 20 sec

def cartesian_distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

def distance2d(x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

def JMT(start, end, T):
    """
    Calculate the Jerk Minimizing Trajectory that connects the initial state
    to the final state in time T.

    INPUTS

    start - the vehicles start location given as a length three array corresponding to initial values of [s, s_dot, s_double_dot]
    end   - the desired end state for vehicle. Like "start" this is a length three array.
    T     - The duration, in seconds, over which this maneuver should occur.

    OUTPUT: an array of length 6, each value corresponding to a coefficent in the polynomial 
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

    EXAMPLE:
    > JMT( [0, 10, 0], [10, 10, 0], 1)
    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    """
    t2 = T * T
    t3 = t2 * T
    t4 = t2 * t2
    t5 = t3 * t2
    Tmat = np.array( [[t3, t4, t5], [3*t2, 4*t3, 5*t4], [6*T, 12*t2, 20*t3]] )

    Sf = end[0]
    Sf_d = end[1]
    Sf_dd = end[2]
    Si = start[0]
    Si_d = start[1]
    Si_dd = start[2]

    Sfmat = np.array( [[Sf - (Si + Si_d*T + 0.5*Si_dd*T*T)], [Sf_d - (Si_d + Si_dd*T)], [Sf_dd - Si_dd]] )
    alpha = np.linalg.inv(Tmat).dot(Sfmat)
    return (Si, Si_d, 0.5*Si_dd, alpha[0], alpha[1], alpha[2])

class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoint_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.current_pose = None
        self.base_lane = None
        self.base_waypoints = None
        self.num_waypoints = 0
        self.base_waypoint_distances = None
        self.Scoeffs = None
        self.Dcoeffs = None

        rospy.spin()

    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.base_waypoints == None:
            return

        #rospy.loginfo("pose_cb::x:%f,y:%f,z:%f; qx:%f,qy:%f,qz:%f,qw:%f", 
        #    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        #    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        self.current_pose = msg

        # find nearest waypoint
        wp1 = self.getNearestWaypointIndex(self.current_pose)
        self.nearestWaypointIndex = wp1
        #rospy.loginfo("closest waypoint: %d; x,y: (%f,%f)", wp1, *self.get_waypoint_coordinate(wp1))

        # return next n waypoints as a Lane pbject
        waypoints = self.base_waypoints[wp1:(wp1 + LOOKAHEAD_WPS)%self.num_waypoints]
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
        return

    """
        # TODO:

        #1. Calculate Frenet coordinates for current_pose
        car_s, car_d = self.getFrenetCoordinate()
        rospy.loginfo("Frenet: %f, %f", frenet_s, frenet_d)


        #2. Calculate velocity in Frenet space
        if self.Scoeffs == None:
            car_us = 0
            car_as = 0
            car_ud = 0
            car_ad = 0
        else:
            # TODO: calc speed, accel from JMT


        #3. FSM to plan route
        targetSpeed = SPEED_LIMIT
        tr_T = LOOKAHEAD_WPS  * 0.02    # generate points @ 0.02 apart each

        #4. Compute final Frenet coordinate at time Tr (end of trajectory)
        accel = min((targetSpeed - us)*(SPEED_LIMIT/TIME_TO_MAX), (SPEED_LIMIT/TIME_TO_MAX));
        final_s = car_s + car_us * tr_T + (0.5 * accel * tr_T * tr_T) # s=ut+1/2 at^2
        final_vs = car_us + accel * tr_T # v=u+at
        final_d = 0 # keep in lane
        final_vd = (final_d-car_d)/tr_T # v=d/t

        #5. Fit polynomical jerk minimizing trajectory
        self.Scoeffs = JMT((car_s, car_us, car_as), (final_s, final_vs, 0), tr_T)
        self.Dcoeffs = JMT((car_d, car_ud, car_ad ), (final_d, final_vd, 0), tr_T)

        #6. Select points for spline, convert them to map coordinates
        Xpts = []
        Ypts = []
        Tpts = []

        # TODO: select 5 points for spline generation

        #7. Generate splines for X and Y
        x_spline = interpolate.InterpolatedUnivariateSpline(Xpts, Tpts)
        y_spline = interpolate.InterpolatedUnivariateSpline(Ypts, Tpts)

        #7. Generate map coordinate points as fixed time intervals (t=0.02)
        waypoints = []
        for i in range(LOOKAHEAD_WPS):
            p = Waypoint()
            p.pose.pose.position.x = x_spline(i*0.02)
            p.pose.pose.position.y = y_spline(i*0.02)
            p.pose.pose.position.z = 0
            q = self.quaternion_from_yaw(float(wp['yaw']))
            p.pose.pose.orientation = Quaternion(*q)
            p.twist.twist.linear.x = float(self.velocity*0.27778)
            waypoints.append(p)

        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
    """

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

    def getFrenetCoordinate(self):
        # next waypoint for current position
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        roll,pitch,yaw = tf.transformations.euler_from_quaternion((
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w))
        map_x,map_y = self.get_waypoint_coordinate(self.nearestWaypointIndex)
        heading = math.atan2( (map_y-y),(map_x-x) )
        angle = math.fabs(yaw-heading)

        nextWaypoint = self.nearestWaypointIndex
        if angle > math.pi/4:
            nextWaypoint += 1

        prevWaypoint = (nextWaypoint - 1) % self.num_waypoints

        next_x, next_y = self.get_waypoint_coordinate(nextWaypoint)
        prev_x, prev_y = self.get_waypoint_coordinate(prevWaypoint)

        n_x = next_x - prev_x
        n_y = next_y - prev_y

        x_x = x - prev_x
        x_y = y - prev_y

        # find the projection of x onto n
        proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
        proj_x = proj_norm*n_x;
        proj_y = proj_norm*n_y;

        frenet_d = distance2d(proj_x, proj_y, x_x, x_y)

        #see if d value is positive or negative by comparing it to a center point
        center_x = 1000-prev_x
        center_y = 2000-prev_y
        centerToPos = distance2d(center_x, center_y, x_x, x_y)
        centerToRef = distance2d(center_x, center_y, proj_x, proj_y)
        if centerToPos <= centerToRef:
            frenet_d *= -1

        frenet_s = self.base_waypoint_distances[prevWaypoint]
        frenet_s += distance2d(0, 0, proj_x, proj_y)
        return (frenet_s, frenet_d)


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
            for i in range(self.num_waypoints):
                pos2 = self.base_waypoints[i].pose.pose.position
                gap = cartesian_distance(pos1,pos2)
                self.base_waypoint_distances.append(d + gap)
                d += gap
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
