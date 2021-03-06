#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, Float32
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''
RUN_FREQUENCY = 20 #Hz

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # PUBLISH TO:
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.controllerEnabled = False

        # SUBSCRIBERS:
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_command_cb)
        
        self.controller = Controller(max_steer_angle)
        self.control_params = {'target_speed_mps':10, 'current_speed_mps':0, 'turn_z':1, 'sample_time_s':(1./RUN_FREQUENCY)}
        
        self.last_throttle = 0
        self.last_steering = 0
        self.last_brake =0
        
        self.loop()

    def dbw_enabled_cb(self, isEnabled):
        if isEnabled:
            self.controllerEnabled = True  
        else: 
            self.controllerEnabled = False
            
    
    def current_velocity_cb(self, twistMsg):
        self.control_params['current_speed_mps'] = twistMsg.twist.linear.x

    def twist_command_cb(self, twistMsg):
        self.control_params['target_speed_mps'] = twistMsg.twist.linear.x
        self.control_params['turn_z'] = twistMsg.twist.angular.z

    def loop(self):
        rate = rospy.Rate(RUN_FREQUENCY) 
        while not rospy.is_shutdown():
            throttle, brake, steer = self.controller.control(**self.control_params)
            if self.controllerEnabled:
                self.publish(throttle, brake, steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        

        self.last_steering = steer
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        #publishing to the brake will cause issues with the throttle
        #only publish when the brake needs to be applied
        if(brake > 0):
            self.last_brake = brake
            bcmd = BrakeCmd()
            bcmd.enable = True
            bcmd.pedal_cmd_type = BrakeCmd.CMD_PERCENT
            bcmd.pedal_cmd = brake
            self.brake_pub.publish(bcmd)
        else:
            self.last_throttle = throttle
            tcmd = ThrottleCmd()
            tcmd.enable = True
            tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
            tcmd.pedal_cmd = throttle
            self.throttle_pub.publish(tcmd)


if __name__ == '__main__':
    DBWNode()
