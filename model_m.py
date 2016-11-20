from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time
from regulators import RegulatorLQR
from pendulum import InvertedPendulum
import json
import sys
import scipy.io

from cost import RegulationTime, SumOfErrorSquared


def simulate_linear_pendulum(K=None,cost_obj=RegulationTime()):
    if K is not None:
        reg.K = K
    pend.regulator = reg
    y = odeint(pend.x_dot_with_regulator, x0, t)
    cost = cost_obj.cost(y-pend.set_point)
    return cost


if __name__ == '__main__':
    M = 0.5
    m = 0.2
    b = 0.1
    I = 0.006
    l = 0.3

    pend = InvertedPendulum(M, m, b, l, I)
    reg = RegulatorLQR(pend.A, pend.B)
    if len(sys.argv)>1:
        da = json.loads(sys.argv[1])
        reg.K = np.array(da)
        print(reg.K)
    pend.regulator = reg
    x0 = [0, 0, np.pi * 0.85, 0]
    t = np.linspace(0, 10, 1e2)
    dat = dict()
    dat['cost'] = simulate_linear_pendulum()
    scipy.io.savemat('matlab/xxx.mat',dat)
    # f = open('matlab/model_m.json','w')
    # json.dump(cost,f)
    # f.close()


