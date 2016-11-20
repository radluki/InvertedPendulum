from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time

from pendulum import InvertedPendulum

if __name__ == '__main__':
    # pendulum parameters
    M = 1
    m = 0.6
    b = 0.1
    I = 0.06
    l = 0.3

    pend = InvertedPendulum(M, m, b, l, I)
    x0 = [0, 0, np.pi * 0.85, 0]
    t = np.linspace(0, 60, 1e5)

    y = odeint(pend.x_dot, x0, t)

    # angle visualization
    fig1 = plt.figure()
    plt.plot(t, y[:, 2], label='angle')
    plt.legend()
    fig1.show()

    fig3 = plt.figure()

    plt.plot(t, y[:, 0], label='position')
    plt.legend()
    plt.show()