#!/usr/bin/env python
PKG = 'waypoint_updater'
import roslib; roslib.load_manifest(PKG)

import sys
import unittest

class TestWaypointUpdater(unittest.TestCase): 
    def test_run(self):
        self.assertTrue(True, 'Test suite works')

if __name__=='__main__':
    import rostest
    rostest.rosrun(PKG, 'wput', TestWaypointUpdater)