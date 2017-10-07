#!/usr/bin/env python
import rospy
import tf
import cv2
import yaml
import math
import sys

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier, TLClassifierSqueeze, TLClassifierVGG16

# Func to calculate cartesian distance
dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

STATE_COUNT_THRESHOLD = 2

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

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # information received from other ROS nodes
        self.pose = None
        self.camera_image = None
        self.num_waypoints = 0
        self.waypoints = None

        # index of waypoint nearest to car
        self.nearestWaypointIndex = -1

        # stores waypoints of traffic light positions
        self.lights = []
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        # One time initialization of the light positions
        self.light_positions = self.config['light_positions']

        #Put subs after variable inits to avoid race condition (call backs arriving during init)
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.tf_sub = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        rospy.spin()

    # callback to store vehicle's current position
    def pose_cb(self, msg):
        self.pose = msg

    # Waypoint callback - data from /waypoint_loader
    # Expect this to be constant, cache it and dont handle beyond 1st call
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
            self.waypoints = waypoints

            # Unregister from waypoint callback - huge improvement in speed
            self.sub2.unregister()

    def traffic_cb(self, msg):
        # Cache the lights information and unsubscribe from topic
        self.lights = msg.lights
        self.tf_sub.unregister()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    # Update nearest waypoint index by searching nearby values
    # waypoints are sorted, so search can be optimized
    def getNearestWaypointIndex(self, pose):
        # previous nearest point not known, do exhaustive search
        # todo: improve with binary search
        if self.nearestWaypointIndex == -1:
            r = [(dl(wp.wp.pose.pose.position, pose.position), i) for i, wp in enumerate(self.waypoints)]
            return min(r, key=lambda x: x[0])[1]
        # Previous nearest waypoint known, so scan points immediately after (& before)
        else:
            d = dl(self.waypoints[self.nearestWaypointIndex].wp.pose.pose.position, pose.position)
            # scan right
            i = self.nearestWaypointIndex
            d1 = d
            found = False
            while True:
                i = (i + 1) % self.num_waypoints
                d2 = dl(self.waypoints[i].wp.pose.pose.position, pose.position)
                if d2 > d1:
                    break
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
                d2 = dl(self.waypoints[i].wp.pose.pose.position, pose.position)
                if d2 > d1:
                    break
                d1 = d2
                found = True
            if found:
                return i+1

            # keep prev value
            return self.nearestWaypointIndex

    # Given the current position of the car, find the closest traffic light
    # and return the index of the light.
    # Assumption: self.lights is updated with the latest upcoming lights
    # NOTE: This function is called only when the light state detected is RED/GREEN/YELLOW
    def getNearestTrafficPoseIndex(self, pose):
        # By default return None
        closest_light_index = None

        if self.lights != []:
            # Set to the max value
            closest_distance = sys.maxint

            # Parse through all the lights
            for index, light in enumerate(self.lights):
                # Calculate the distance from the light to the car
                distance_from_light = dl(light.pose.pose.position, pose.position)

                # Check if the closest light is found
                if distance_from_light < closest_distance:
                    # Update the closest distanc
                    closest_distance = distance_from_light
                    # Update the corresponding index
                    closest_light_index = index

        # Return the index of the closest light
        return closest_light_index

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # First get the closest light to the car based on the current car's pose
        closest_light_index = self.getNearestTrafficPoseIndex(pose)

        # Get the pose of the closest traffic light and then find
        # the nearest way point index to the light
        traffic_light_pose = self.lights[closest_light_index].pose.pose

        # Set the nearestWaypointIndex
        self.nearestWaypointIndex = self.getNearestWaypointIndex(traffic_light_pose)

        # Add debug
        rospy.loginfo("WP Index closest to the light is %d", self.nearestWaypointIndex)

        return self.nearestWaypointIndex

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        state = self.get_light_state()

        # Only get the closest way point index if the light state is RED/GREEN/YELLOW
        if self.pose and state != TrafficLight.UNKNOWN:
            # Compute the closest way point index
            self.get_closest_waypoint(self.pose.pose)

            # NOTE: self.nearestWaypointIndex is set in the call above
            # Return the state and the nearestWaypointIndex
            return self.nearestWaypointIndex, state
        else:
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
