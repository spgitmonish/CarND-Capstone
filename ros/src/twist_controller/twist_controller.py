from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_SPEED_MPH = 50
BRAKE_TORQUE_MAX = 3412

class Controller(object):
    def __init__(self, *args, **kwargs):
        #accelerator PID
        self.gainp=.15
        self.gaini=.006
        self.gaind=.013
        #pid will use m/s, all velocities must be converted from mph to m/s
        self.max_speed_mps = MAX_SPEED_MPH*ONE_MPH
        self.min_speed_mps = 0
        
        self.steerp=2.1
        self.steeri=.1
        self.steerd=.8
        self.speed_pid = PID(self.gainp,self.gaini,self.gaind, -1.0 , 1.0)
        self.steer_pid = PID(self.steerp,self.steeri,self.steerd, -1*args[0], args[0]) 
        
        self.throttle = 0.
        self.brake = 0.
        self.steer = 0.
        
    def control(self, target_speed_mps, current_speed_mps, sample_time_s=.1, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        effort = self.speed_pid.step(target_speed_mps-current_speed_mps, sample_time_s)
        if effort < 0:
            self.throttle = 0
            self.brake = abs(effort) * BRAKE_TORQUE_MAX
        else:
            self.throttle = effort
            self.brake = 0
        if 'turn_z' in kwargs.keys(): 
            self.steer = self.steer_pid.step(kwargs['turn_z'], sample_time_s)
                    
        return self.throttle, self.brake, self.steer
