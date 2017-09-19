#!/usr/bin/env python
PKG = 'waypoint_updater'
import roslib; roslib.load_manifest(PKG) #not needed with catkin

import sys
import unittest
from waypoint_updater import WaypointUpdater

class TestWaypointUpdater(unittest.TestCase): 
    def test_run(self):
        self.assertTrue(True, 'Test suite works')
        
    def test_makeWaypointUpdater(self):
        wu = WaypointUpdater()
        self.assertIsNotNone(wu, 'Failed to init waypoint updater')
        
    def test_initialState(self):
        wu = WaypointUpdater()
        self.assertEqual(wu.fsm_state, 0, 'Init WU be in state FSM_GO(0)') 

if __name__=='__main__':
    import rostest
    rostest.rosrun(PKG, 'wput', TestWaypointUpdater)
