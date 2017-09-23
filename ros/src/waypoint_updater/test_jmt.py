import numpy as np

def JMT(start, end, T):
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

    Sfmat = np.array( [[Sf - (Si + Si_d*T + 0.5*Si_dd*T*T)], [Sf_d - (Si_d + Si_dd*T)], [Sf_dd - Si_dd]] )

    alpha = np.linalg.inv(Tmat).dot(Sfmat)

    return (Si, Si_d, 0.5*Si_dd, alpha[0], alpha[1], alpha[2])


if __name__ == '__main__':
    print(JMT([0, 10, 0], [10,10,0], 1))
    print(JMT([0, 10, 0], [20,15,20], 2))
    print(JMT([5, 10, 2], [-30,-20,-4], 5))

