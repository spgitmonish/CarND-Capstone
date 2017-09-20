#!/usr/bin/env python
import math

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