#!/usr/bin/env python
import math
import numpy as np

def clip(val, min, max):
    if val < min:
        return min
    if val > max:
        return max
    return val

def cartesian_distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

def distance2d(x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

def theta_slope(w1, w2):
    x1 = w1.pose.pose.position.x
    y1 = w1.pose.pose.position.y
    x2 = w2.pose.pose.position.x
    y2 = w2.pose.pose.position.y
    return math.atan2((y2-y1), (x2-x1))

def XYGlobalToLocal(xglobal,yglobal,car_x,car_y,car_yaw):
    shift_x = xglobal - car_x
    shift_y = yglobal - car_y
    x = shift_x*cos(0-car_yaw)-shift_y*sin(0-car_yaw)
    y = shift_x*sin(0-car_yaw)+shift_y*cos(0-car_yaw)
    return x, y

def XYLocalToGlobal(xlocal, ylocal, car_x, car_y, car_yaw):
    x = xlocal*cos(car_yaw)-ylocal*sin(car_yaw)+car_x
    y = xlocal*sin(car_yaw)+ylocal*cos(car_yaw)+car_y
    return x, y

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