#!/usr/bin/env python

import rospy
import csv
import tf
from geometry_msgs.msg import Pose, Point, PoseStamped, TwistStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
from scipy.interpolate import interp1d, splrep, splev

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''
TIME_PERIOD_PUBLISHED = 2. #sec
LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
TIME_STEP = TIME_PERIOD_PUBLISHED / LOOKAHEAD_WPS
SPEED_LIMIT = 10.0 # m/s
TIME_TO_MAX = 5.0 # 0 to 50 in 20 sec
LIGHT_BREAKING_DISTANCE_METERS = 30 # meters

FSM_GO = 0
FSM_STOPPING = 1

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

    Sfmat = np.array( [Sf - (Si + Si_d*T + 0.5*Si_dd*T*T), Sf_d - (Si_d + Si_dd*T), Sf_dd - Si_dd] )
    alpha = np.linalg.inv(Tmat).dot(Sfmat)
    
    #return (Si, Si_d, 0.5*Si_dd, alpha[0], alpha[1], alpha[2])
    return (alpha[2], alpha[1], alpha[0], 0.5*Si_dd, Si_d, Si)

class WaypointInfo(object):
    def __init__(self, wp_obj, dist_to_next, frenet_s):
        self.wp = wp_obj
        self.width = dist_to_next
        self.s = frenet_s
        self.t = 0.

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
        self.waypoints = None
        self.num_waypoints = 0
        self.nearestWaypointIndex = -1
        self.Scoeffs = None
        self.Dcoeffs = None
        self.output = None
        # self.logfile = open('path.csv', 'wb')

        """
        light 1: closest waypoint: 289; x,y: (1145.720000,1184.640000)
        light 2: closest waypoint: 750; x,y: (1556.820000,1158.860000)

        """
        self.traffic_light = 750

        # initial state machine state
        self.fsm_state = FSM_GO

        rospy.spin()

    def traffic_lights_off(self, timer_event):
        rospy.loginfo("traffic_light_cleared")
        self.traffic_light = None

    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.waypoints == None:
            return

        #rospy.loginfo("pose_cb::x:%f,y:%f,z:%f; qx:%f,qy:%f,qz:%f,qw:%f", 
        #    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        #    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        self.current_pose = msg

        # find nearest waypoint
        wp_start = self.getNearestWaypointIndex(self.current_pose)
        self.nearestWaypointIndex = wp_start
        #rospy.loginfo("closest waypoint: %d; x,y: (%f,%f)", wp_start, *self.get_waypoint_coordinate(wp_start))

        # return next n waypoints as a Lane pbject
        if self.fsm_state == FSM_GO:
            # Do we have a traffic light in the vicinity ?
            if self.traffic_light != None and self.distanceToWaypoint(self.traffic_light) < LIGHT_BREAKING_DISTANCE_METERS:
                rospy.loginfo("slowing for traffic light: %d => %d", wp_start, self.traffic_light)
                self.fsm_state = FSM_STOPPING
                # simulates a 30 - second stop
                rospy.Timer(rospy.Duration.from_sec(20), self.traffic_lights_off, oneshot=True)
                waypoints = self.decelarate(self.traffic_light)
            else:
                # continue FSM_GO
                waypoints = self.accelarate()


        else: # self.fsm_state == FSM_STOPPING

            # wait for light to go off
            if self.traffic_light != None:
                # continue decelarating
                waypoints = self.decelarate(self.traffic_light)
            else:
                # back to FSM_GO
                self.fsm_state = FSM_GO
                waypoints = self.accelarate()
        
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
        return

    def accelarate(self):
        # accelaration
        #rospy.loginfo("accelarating: %f", a)
        a = clip((SPEED_LIMIT - self.velocity) / TIME_PERIOD_PUBLISHED, -SPEED_LIMIT/TIME_TO_MAX, SPEED_LIMIT/TIME_TO_MAX)
        return self.jmt_interpolate_waypoints(a)

    def decelarate(self, wp_end):
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
        u = self.velocity
        s = self.distanceToWaypoint(wp_end)
        a = -u**2 / (2*s)
        rospy.loginfo("decelarate: %f", a)

        return self.jmt_interpolate_waypoints(min(-SPEED_LIMIT/TIME_TO_MAX, a))


    def jmt_interpolate_waypoints(self, a):

        T = []
        X = []
        Y = []

        if self.Scoeffs == None or self.Dcoeffs == None:
            s, d = self.getFrenetCoordinate()
            su = 0
            sa = 0
            du = 0
            da = 0
            
            T.append(0)
            X.append(self.current_pose.pose.position.x)
            Y.append(self.current_pose.pose.position.y)
        else:
            # where are we in time in relation to previously generated polynomial
            r = [(distance2d(wp.pose.pose.position.x, wp.pose.pose.position.y, self.current_pose.pose.position.x, self.current_pose.pose.position.y), i) for i,wp in enumerate(self.output)]
            dist_arr = sorted(r, key=lambda x: x[0])
            time_elapsed = dist_arr[0][1] * TIME_STEP
            #t = min(r, key=lambda x: x[0])[1] * TIME_STEP
            # self.logfile.write("Time elapsed: {}, {}\n".format(time_elapsed, dist_arr[0][1]))

            #s,d = self.getFrenetCoordinate()
            s = np.polyval(self.Scoeffs, time_elapsed)
            su = np.polyval(np.polyder(self.Scoeffs,1), time_elapsed)
            sa = np.polyval(np.polyder(self.Scoeffs,2), time_elapsed)
            d = np.polyval(self.Dcoeffs, time_elapsed)
            du = np.polyval(np.polyder(self.Dcoeffs,1), time_elapsed)
            da = np.polyval(np.polyder(self.Dcoeffs,2), time_elapsed)

            T.append(0)
            X.append(self.current_pose.pose.position.x)
            Y.append(self.current_pose.pose.position.y)

        #self.logfile.write("a={}, s={}, d={}, x={}, y={}\n".format(a, s, d, self.current_pose.pose.position.x, self.current_pose.pose.position.y))

        t = TIME_PERIOD_PUBLISHED
        f_s = s + clip((su * t + 0.5 * a * t * t), 0, 100)
        f_sv = clip(su + a * t, 0, SPEED_LIMIT)
        f_sa = 0

        f_d = 0
        f_dv = clip(d / (4*t), 0, SPEED_LIMIT)
        f_da = 0

        self.Scoeffs = JMT( [s,su,sa], [f_s, f_sv, f_sa], t )
        self.Dcoeffs = JMT( [d,du,da], [f_d, f_dv, f_da], t )

        # self.logfile.write("JMT Inputs: s={}, su={}, sa={}, f_s={}, f_sv={}, f_sa={}\n".format(s,su,sa,f_s, f_sv, f_sa))
        # self.logfile.write("JMT Inputs: d={}, du={}, da={}, f_d={}, f_dv={}, f_da={}\n".format(d,du,da,f_d, f_dv, f_da))

        # self.logfile.write("Scoeffs={}, Dcoeffs={}\n".format(self.Scoeffs, self.Dcoeffs))
        for t in np.arange(TIME_PERIOD_PUBLISHED / 4., TIME_PERIOD_PUBLISHED * 2, TIME_PERIOD_PUBLISHED/4.):
            s_t = np.polyval(self.Scoeffs, t)
            d_t = np.polyval(self.Dcoeffs, t)
            try:
                x, y, heading = self.frenet2XY(s_t, d_t)
            except ValueError:
                rospy.loginfo("s:%f, d:%f",s, d)
            T.append(t)
            X.append(x)
            Y.append(y)


        # k = 5 # qunitic
        # if len(T) < 6:
        #     k = 3 # cubic
        # if len(T) < 4:
        #     k = 1 # linear

        try:
            X_spline = splrep(T, X, k = 1) 
            Y_spline = splrep(T, Y, k = 1)
        except ValueError:
            # self.logfile.write("Spline on T: {} \n".format(T))
            # self.logfile.write("Spline on X: {} \n".format(X))
            # self.logfile.write("Spline on Y: {} \n".format(Y))

        self.output = []
        for t in np.arange(0, TIME_PERIOD_PUBLISHED, TIME_PERIOD_PUBLISHED / LOOKAHEAD_WPS ):
            p = Waypoint()
            p.pose.pose.position.x = round(float(splev(t, X_spline)),1)
            p.pose.pose.position.y = round(float(splev(t, Y_spline)),1)
            p.twist.twist.linear.x = float(clip(su + a*t, 0, SPEED_LIMIT))
            self.output.append(p)

        # dump output to csv
        # spamwriter = csv.writer(self.logfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # for o in self.output:
        #     spamwriter.writerow([o.pose.pose.position.x, o.pose.pose.position.y, o.twist.twist.linear.x])
        # spamwriter.writerow(["-----"]*3)

        return self.output


    def simple_interpolate_waypoints(self, a):
        output = []
        v = self.velocity
        wp = self.nearestWaypointIndex
        theta = theta_slope(self.waypoints[wp % self.num_waypoints].wp, self.waypoints[(wp+1) % self.num_waypoints].wp)

        for t in np.arange(0, LOOKAHEAD_WPS*0.02, 0.02):
            r = v*t + 0.5*a*(t**2)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p = Waypoint()
            p.pose.pose.position.x = float(self.waypoints[wp % self.num_waypoints].wp.pose.pose.position.x + x)
            p.pose.pose.position.y = float(self.waypoints[wp % self.num_waypoints].wp.pose.pose.position.y + y)
            p.pose.pose.position.z = self.waypoints[wp % self.num_waypoints].wp.pose.pose.position.z
            p.pose.pose.orientation = self.waypoints[wp % self.num_waypoints].wp.pose.pose.orientation
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

    # Velocity is passed in as m/s
    def velocity_cb(self, vel):
        self.velocity = vel.twist.linear.x
        #rospy.loginfo("velocity: %f", vel.twist.linear.x)

    # Waypoint callback - data from /waypoint_loader
    # I expect this to be constant, so we cache it and dont handle beyond 1st call
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

        frenet_s = self.waypoints[prevWaypoint].s
        frenet_s += distance2d(0, 0, proj_x, proj_y)

        #rospy.loginfo("x=%f, y=%f => s=%f, d=%f: nearest: %d, map_x:%f, map_y:%f, yaw:%f, heading:%f", 
        #    self.current_pose.pose.position.x, self.current_pose.pose.position.y, frenet_s, frenet_d, 
        #    self.nearestWaypointIndex, map_x, map_y, yaw, heading)

        return (frenet_s, frenet_d)

    def frenet2XY(self, s,d):

        wp1 = min([(s-o.s, i) for i,o in enumerate(self.waypoints) if s >= o.s], key=lambda x: x[0])[1]
        wp2 = (wp1+1)%self.num_waypoints

        wp1_x = self.waypoints[wp1].wp.pose.pose.position.x
        wp2_x = self.waypoints[wp2].wp.pose.pose.position.x
        wp1_y = self.waypoints[wp1].wp.pose.pose.position.y
        wp2_y = self.waypoints[wp2].wp.pose.pose.position.y

        heading = math.atan2(wp2_y - wp1_y, wp2_x - wp1_x)

        seg_s = s - self.waypoints[wp1].s

        seg_x = wp1_x+seg_s*math.cos(heading)
        seg_y = wp1_y+seg_s*math.sin(heading)

        perp_heading = heading - math.pi/2

        x = seg_x + d*math.cos(perp_heading);
        y = seg_y + d*math.sin(perp_heading);

        return (x, y, heading);


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
