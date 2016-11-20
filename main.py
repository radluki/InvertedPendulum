from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import fmin_bfgs,minimize

from regulators import RegulatorLQR
from pendulum import InvertedPendulum
from cost import RegulationTime, SumOfErrorSquared

from swing_up_test import plot_state

def simulate_linear_pendulum(K=None,cost_obj=RegulationTime()):
    if K is not None:
        reg.K = K
    pend.regulator = reg
    y = odeint(pend.x_dot_with_regulator, x0, t)
    cost = cost_obj.cost(y-pend.set_point)
    return cost


if __name__=='__main__':
    # pendulum parameters
    M = 0.5
    m = 0.2
    b = 0.1
    I = 0.006
    l = 0.3

    pend = InvertedPendulum(M, m, b, l, I)
    Q = np.eye(4)
    #Q[2, 2] *= 1e3
    R = np.eye(1) * 1e0
    reg = RegulatorLQR(pend.A, pend.B, Q=Q, R=R)
    pend.regulator = reg
    x0 = [0, 0, np.pi * 0.01, 0]
    t = np.linspace(0, 10, 1e3)

    print("sim:", simulate_linear_pendulum())
    t1 = time.time()
    K = fmin_bfgs(simulate_linear_pendulum, reg.K, gtol=1e-3)
    t1 = time.time() - t1
    print("Czas:", t1)
    print("sim:", simulate_linear_pendulum(K))


    print("Proportional regulator:",reg.K)
    t0 = time.time()
    y = odeint(pend.x_dot_with_regulator, x0, t)
    t1 = time.time() - t0
    print("Time of solving ode:",t1)

    plot_state(t,y,pend)



