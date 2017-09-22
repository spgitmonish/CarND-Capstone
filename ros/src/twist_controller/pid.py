
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.prev_cte = self.i_error = self.d_error = 0.
        self.n = 0
        self.mse = 0.

    def reset(self):
        self.prev_cte = self.i_error = self.d_error = 0.
        self.n = 0
        self.mse = 0.

    def update(self, cte):
        self.i_error += cte
        self.mse += cte*cte
        self.d_error = cte - self.prev_cte
        self.prev_cte = cte
        self.n += 1

    def compute(self, cte):
        return -self.kp * cte - self.kd * self.d_error - self.ki * self.i_error
