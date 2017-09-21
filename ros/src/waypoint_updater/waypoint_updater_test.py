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

FSM = {'STOP':0,
        'GO':1,
        'STOPPING':2}

class TestWaypointUpdater(unittest.TestCase):
    def test_run(self):
        self.assertTrue(True, 'Test suite works')
        
    def test_makeWaypointUpdater(self):
        wu = WaypointUpdater()
        self.assertIsNotNone(wu, 'Failed to init waypoint updater')
        
    def test_initialState(self):
        wu = WaypointUpdater()
        self.assertEqual(wu.fsm_state, FSM['GO'], 'Init WU be in state FSM_GO(1)')
        
    def test_update_fsm(self):
        wu = WaypointUpdater()
        self.assertEqual(wu.fsm_state, FSM['GO'], 'Init WU be in state FSM_GO(1)')
        wu.fsm_state = wu.update_fsm(FSM['GO'],20)
        self.assertEqual(wu.fsm_state, FSM['STOPPING'], 'fsm slow when approaching light')
        wu.fsm_state = wu.update_fsm(FSM['STOPPING'],0)
        self.assertEqual(wu.fsm_state, FSM['STOP'], 'fsm stopped when at light')
        wu.fsm_state = wu.update_fsm(FSM['STOP'],100)
        self.assertEqual(wu.fsm_state, FSM['GO'], 'fsm goes when new traffic light distance')        
        

if __name__=='__main__':
    import rostest
    rostest.rosrun(PKG, 'wput', TestWaypointUpdater)
