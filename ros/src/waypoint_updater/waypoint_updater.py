#!/usr/bin/env python

import math
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from eventlet import event
import rospy
import tf

from geometry_msgs.msg import Pose, Point, PoseStamped, TwistStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint

import geometry_utils

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

#FSM_GO = 0
#FSM_STOPPING = 1

#finite states
FSM = {'STOP':0,
        'GO':1,
        'STOPPING':2}

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

        """
        light 1: closest waypoint: 289; x,y: (1145.720000,1184.640000)
        light 2: closest waypoint: 750; x,y: (1556.820000,1158.860000)

        """
        self.traffic_light = 750
        # initial state machine state
        self.fsm_state = FSM['GO']

        rospy.spin()


    # called when car's pose has changed
    # respond by emitting next set of final waypoints
    def pose_cb(self, msg):
        if self.waypoints == None:
            return

        self.current_pose = msg

                # find nearest waypoint
        wp_start = self.getNearestWaypointIndex(self.current_pose)
        self.nearestWaypointIndex = wp_start
        self.traffic_light = self.update_traffic_light(wp_start, self.traffic_light)
        
        distance = -1 
        if self.traffic_light != None:
            distance = self.distanceToWaypoint(self.traffic_light)
        self.fsm_state = self.update_fsm(self.fsm_state,distance)
        
        self.update_waypoints()

    def update_traffic_light(self, nearest_wp, tf_wp):
        if nearest_wp > tf_wp:
            return None
        
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
            if self.traffic_light == None or tf_distance > 10 or tf_distance < -10:
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
        
    def make_target_speed_wp(self):
        wp = self.nearestWaypointIndex
        theta = geometry_utils.theta_slope(self.base_waypoints[wp % self.num_waypoints], self.base_waypoints[(wp+1) % self.num_waypoints])
        v = self.velocity
        a = geometry_utils.clip((SPEED_LIMIT - self.velocity) / TIME_PERIOD_PUBLISHED, -SPEED_LIMIT/TIME_TO_MAX, SPEED_LIMIT/TIME_TO_MAX)
        output = []
        for t in np.arange(0, LOOKAHEAD_WPS*0.1, 0.1):
            r = v*t + 0.5*a*(t**2)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p = Waypoint()
            p.pose.pose.position.x = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.x + x)
            p.pose.pose.position.y = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.y + y)
            p.pose.pose.position.z = self.base_waypoints[wp % self.num_waypoints].pose.pose.position.z
            p.pose.pose.orientation = self.base_waypoints[wp % self.num_waypoints].pose.pose.orientation
            p.twist.twist.linear.x = float(geometry_utils.clip(v+a*t,0,SPEED_LIMIT))
            output.append(p)
        return output
    
    def make_slow_speed_wp(self):
        wp = self.nearestWaypointIndex
        theta = geometry_utils.theta_slope(self.base_waypoints[wp % self.num_waypoints], self.base_waypoints[(wp+1) % self.num_waypoints])
        v = self.velocity
        a = geometry_utils.clip((0 - self.velocity) / TIME_PERIOD_PUBLISHED, -SPEED_LIMIT/TIME_TO_MAX, SPEED_LIMIT/TIME_TO_MAX)
        output = []
        t_inc = .1
        for t in np.arange(0, LOOKAHEAD_WPS*t_inc, t_inc):
            r = v*t + 0.5*a*(t**2)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p = Waypoint()
            p.pose.pose.position.x = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.x + x)
            p.pose.pose.position.y = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.y + y)
            p.pose.pose.position.z = self.base_waypoints[wp % self.num_waypoints].pose.pose.position.z
            p.pose.pose.orientation = self.base_waypoints[wp % self.num_waypoints].pose.pose.orientation
            p.twist.twist.linear.x = float(geometry_utils.clip(v+a*t,0,SPEED_LIMIT))
            output.append(p)
        return output

    
    def accelerate(self):
        # accelaration
        #rospy.loginfo("accelarating: %f", a)
        a = geometry_utils.clip((SPEED_LIMIT - self.velocity) / TIME_PERIOD_PUBLISHED, -SPEED_LIMIT/TIME_TO_MAX, SPEED_LIMIT/TIME_TO_MAX)
        return self.jmt_interpolate_waypoints(a)

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
        u = self.velocity
        s = self.distanceToWaypoint(wp_end)
        a = -u**2 / (2*s)
        rospy.loginfo("decelarate: %f", a)

        return self.jmt_interpolate_waypoints(a)


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
            r = [(geometry_utils.distance2d(wp.pose.pose.position.x, wp.pose.pose.position.y, self.current_pose.pose.position.x, self.current_pose.pose.position.y), i) for i,wp in enumerate(self.output)]
            time_elapsed = min(r, key=lambda x: x[0])[1] * TIME_STEP

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
        f_s = s + geometry_utils.clip((su * t + 0.5 * a * t * t), 0, 100)
        f_sv = geometry_utils.clip(su + a * t, 0, SPEED_LIMIT)
        f_sa = 0

        f_d = 0
        f_dv = geometry_utils.clip(d / (4*t), 0, SPEED_LIMIT)
        f_da = 0

        self.Scoeffs = geometry_utils.JMT( [s,su,sa], [f_s, f_sv, f_sa], t )
        self.Dcoeffs = geometry_utils.JMT( [d,du,da], [f_d, f_dv, f_da], t )

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
            pass
            # self.logfile.write("Spline on T: {} \n".format(T))
            # self.logfile.write("Spline on X: {} \n".format(X))
            # self.logfile.write("Spline on Y: {} \n".format(Y))

        self.output = []
        for t in np.arange(0, TIME_PERIOD_PUBLISHED, TIME_PERIOD_PUBLISHED / LOOKAHEAD_WPS ):
            p = Waypoint()
            p.pose.pose.position.x = round(float(splev(t, X_spline)),1)
            p.pose.pose.position.y = round(float(splev(t, Y_spline)),1)
            p.twist.twist.linear.x = float(geometry_utils.clip(su + a*t, 0, SPEED_LIMIT))
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
            p.twist.twist.linear.x = float(geometry_utils.clip(v+a*t,0,SPEED_LIMIT))
            output.append(p)
        return output

    def spline_interpolate_waypoints(self, a):
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
            theta = geometry_utils.theta_slope(self.base_waypoints[wp % self.num_waypoints], self.base_waypoints[(wp+1) % self.num_waypoints])

            for t in np.arange(0, LOOKAHEAD_WPS*0.02, 0.02):
                r = v*t + 0.5*a*(t**2)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                p = Waypoint()
                p.pose.pose.position.x = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.x + x)
                p.pose.pose.position.y = float(self.base_waypoints[wp % self.num_waypoints].pose.pose.position.y + y)
                p.pose.pose.position.z = self.base_waypoints[wp % self.num_waypoints].pose.pose.position.z
                p.pose.pose.orientation = self.base_waypoints[wp % self.num_waypoints].pose.pose.orientation
                p.twist.twist.linear.x = float(geometry_utils.clip(v+a*t,0,SPEED_LIMIT))
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
            p.twist.twist.linear.x = float(geometry_utils.clip(v+a*t,0,SPEED_LIMIT))
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
                gap = geometry_utils.distance2d(wp1_x, wp1_y, wp2_x, wp2_y)
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

        frenet_d = geometry_utils.distance2d(proj_x, proj_y, x_x, x_y)

        #see if d value is positive or negative by comparing it to a center point
        center_x = 1000-prev_x
        center_y = 2000-prev_y
        centerToPos = geometry_utils.distance2d(center_x, center_y, x_x, x_y)
        centerToRef = geometry_utils.distance2d(center_x, center_y, proj_x, proj_y)
        if centerToPos <= centerToRef:
            frenet_d *= -1

        frenet_s = self.waypoints[prevWaypoint].s
        frenet_s += geometry_utils.distance2d(0, 0, proj_x, proj_y)

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
