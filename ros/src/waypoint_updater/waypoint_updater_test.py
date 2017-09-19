#!/usr/bin/env python
from rospy.timer import sleep
PKG = 'waypoint_updater'
import roslib; roslib.load_manifest(PKG) #not needed with catkin

import sys
import unittest
from waypoint_updater import WaypointUpdater

import rospy
import tf
from geometry_msgs.msg import Pose, Point, PoseStamped, TwistStamped, Quaternion
from styx_msgs.msg import Lane, Waypoint

class TestWaypointUpdater(unittest.TestCase):
    def test_run(self):
        self.assertTrue(True, 'Test suite works')
        
    def test_makeWaypointUpdater(self):
        wu = WaypointUpdater()
        self.assertIsNotNone(wu, 'Failed to init waypoint updater')
        
    def test_initialState(self):
        wu = WaypointUpdater()
        self.assertEqual(wu.fsm_state, 1, 'Init WU be in state FSM_GO(1)')

if __name__=='__main__':
    import rostest
    rostest.rosrun(PKG, 'wput', TestWaypointUpdater)
