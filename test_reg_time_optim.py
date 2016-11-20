import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class PID(object):

    def __init__(self,p,i,d):
        self.p = p
        self.i = i
        self.d = d

    def control(self,x,sp):
        e = sp-x
        return self.p*e

class InerObj(object):

    def __init__(self,T,k,set_point=1,reg=None):
        self.T = T
        self.k = k
        self.reg = reg
        self.set_point = set_point

    def xdot(self,x,t,u=1):
        return -1/self.T*x + u*self.k/self.T

    def xdot_with_reg(self,x,t):
        u = self.reg.control(x,self.set_point)
        return self.xdot(x,t,u+1)


class InerPI(object):
    """Inertial object with regulator PI"""
    def __init__(self,T,k,p,Ti):
        self.T = T
        self.k = k
        self.p = p
        self.Ti = Ti
        self.A = None
        self.C = None
        self.update([self.p,self.Ti])

    def update(self,x):
        self.p = x[0]
        self.Ti = x[1]
        a2 = self.Ti*self.T
        a1 = (self.Ti+self.k*self.p*self.Ti)/a2
        a0 = self.k/a2
        b1 = self.k*self.p*self.Ti/a2
        b0 = self.k/a2
        self.A = np.array([[0,1],[-a0,-a1]])
        self.C = np.array([[b0,b1]])

    def xdot(self,x,t):
        return self.A.dot(x) +np.array([0,1])

    def simulate(self,t):
        x0 = [0,0]
        x = odeint(self.xdot,x0,t)
        y = [self.C.dot(k) for k in x]
        return y


if __name__ == "__main__":
    # iner = InerObj(4,2,reg=PID(10,0,0))
    # t = np.linspace(0,20,1e3)
    # x0 = 0
    # y = odeint(iner.xdot_with_reg,x0,t)

    t = np.linspace(0, 20, 1e3)
    iner = InerPI(10,2,10,0.0005)
    y = iner.simulate(t)

    plt.plot(t,y)
    plt.show()