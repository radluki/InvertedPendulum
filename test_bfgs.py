from scipy.optimize import fmin_bfgs,minimize
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time

def fun(x):
    return x[0]**2 + 3*x[1]**2

if __name__=='__main__':
    t0 = time.time()
    x = fmin_bfgs(fun,[12,22])
    t1 = time.time() - t0

    t0 = time.time()
    x2 = minimize(fun,[12,22])
    t2 = time.time() - t0

    print("Time:",t1)
    print("Time2:", t2)
    print(x)
    print(x2)