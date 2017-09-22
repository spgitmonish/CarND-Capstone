from pid import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_SPEED_MPH = 50
BRAKE_TORQUE_MAX = 3412.

class Controller(object):
    def __init__(self, *args, **kwargs):
        #accelerator PID
        # TODO: Tune numbers
        self.gainp=50
        self.gaini=0
        self.gaind=0
        #pid will use m/s, all velocities must be converted from mph to m/s
        self.max_speed_mps = MAX_SPEED_MPH*ONE_MPH
        self.min_speed_mps = 0
        
        # TODO: Tune numbers
        self.steerp=-2
        self.steeri=0
        self.steerd=0
        self.speed_pid = PID(self.gainp,self.gaini,self.gaind, -1.0 , 1.0)
        self.steer_pid = PID(self.steerp,self.steeri,self.steerd, -1*args[0], args[0]) 
        
        self.throttle = 0.
        self.brake = 0.
        self.steer = 0.
        
    def control(self, target_speed_mps, current_speed_mps, sample_time_s=.1, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        cte = target_speed_mps - current_speed_mps
        throttle = self.speed_pid.compute(cte)
        self.speed_pid.update(cte)
        #rospy.loginfo("target_speed_mps: %f, current_speed_mps: %f, cte: %f, throttle: %f", target_speed_mps, current_speed_mps, cte, throttle)

        if throttle > 0:
            self.throttle = 0
            self.brake = abs(throttle) * 10
        else:
            self.throttle = abs(throttle)
            self.brake = 0

        if 'turn_z' in kwargs.keys(): 
            cte = kwargs['turn_z']
            self.steer = self.steer_pid.compute(cte)
            self.steer_pid.update(cte)
                    
        return self.throttle, self.brake, self.steer
