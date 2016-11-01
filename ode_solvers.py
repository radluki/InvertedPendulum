from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix
import time
from scipy.linalg import solve_continuous_are
from scipy.optimize import fmin_bfgs,basinhopping


class InvertedPendulum(object):

    def __init__(self,M,m,b,l,I,regulator=None,set_point=[0,0,np.pi,0],g = 9.81):
        """
        Creates model of inverted pendulum with cart
        :param M: cart mass
        :param m: pendulum mass
        :param b: friction parameter
        :param l: ditance between point of rotation and center of gravity of puendulum
        :param I: moment of inertia of pendulum
        :param g: gravitational acceleration
        """
        # phisical parameters
        self.M = M
        self.m = m
        self.b = b
        self.l = l
        self.I = I
        self.g = g
        # computational parameters
        self.mat1 = np.array([[M + m, 0],[0,I+m*l**2]])
        self.mat2 = np.array([[0,m*l],[m*l,0]])
        self.linearize()
        self.regulator = regulator
        self.set_point = np.array(set_point)

    def x_dot(self,x,t=0,F=0):
        # x0 = x[0]
        x1 = x[1]
        theta0 = x[2]
        theta1 = x[3]

        x2,theta2 = np.linalg.inv(self.mat1 + self.mat2*np.cos(theta0)).dot(\
            [-self.b*x1 + self.m*self.l*theta1**2*np.sin(theta0) + F,\
             -self.m*self.g*self.l*np.sin(theta0)])

        x_prime = [x1,x2,theta1,theta2]
        return x_prime

    def linearize(self, up_down ='up'):
        if up_down == 'up':
            theta = np.pi
        elif up_down == 'down':
            theta = 0
        else:
            raise Exception('up or down')
        A = np.zeros((4,4))
        B = [0,1,0,0]
        A[0,0] = 1
        A[3,2] = 1
        A[1:3,1::2] = self.mat1 + self.mat2*np.cos(theta)
        A = np.linalg.inv(A)
        B = A.dot(B)
        A[:,1] = A[:,0] + (-self.b)*A[:,1]
        A[:,2] *=  (-self.m*self.g*self.l*np.cos(theta))
        #A[:,3] *= 1
        A[:,0] = 0
        B = np.reshape(B, (4, 1))
        self.A = A
        self.B = B


    def x_dot_linear_regulator(self,x,t):
        F = self.regulator.control(x,self.set_point)
        x_dot = self.x_dot(x,t,F)

        return x_dot

class LinearReulator(object):

    def __init__(self,K):
        self.K = K

    def control(self,x,set_point):
        dx = [a-b for a,b in zip(x,set_point)]
        return self.K.dot(dx)

class RegulatorLQR(LinearReulator):

    def __init__(self,A,B,Q=np.eye(4),R=np.eye(1)):
        X = solve_continuous_are(A, B, Q, R)
        K = -np.linalg.inv(R)*B.T.dot(X)
        self.K = K




# pendulum parameters
M = 0.5
m = 0.2
b = 0.1
I = 0.006
l = 0.3


pend = InvertedPendulum(M, m, b, l, I)
reg = RegulatorLQR(pend.A,pend.B)
pend.regulator = reg
x0 = [0,0,np.pi*0.85,0]
t = np.linspace(0,10,1e3)

class RegulationTime(object):
    def cost(self,e):
        for i in reversed(range(len(e))):

            print(i)
            if np.abs(e[i,2]) > 0.01:
                cost = sum(np.abs(e[:i,2]))
                print(t[i])
                print(cost)
                return cost
        return 1e10

class Error2Integral(object):
    def cost(self,e):
        err =  sum(e[:,2]**2)
        #print(err)
        return err

def simulate_linear_pendulum(K=None,cost_obj=RegulationTime()):
    if K is not None:
        reg.K = K
    pend.regulator = reg
    y = odeint(pend.x_dot_linear_regulator, x0, t)
    cost = cost_obj.cost(y-pend.set_point)
    return cost

def f(x):
    return x[0]**2 + 5*x[1]**2
#print(fmin_bfgs(f,[1,2]))

print("sim:",simulate_linear_pendulum())
t1 = time.time()
K = fmin_bfgs(simulate_linear_pendulum,reg.K,maxiter=5)
t1 = time.time() - t1
print("Czas:",t1)
print("sim:",simulate_linear_pendulum(K))
#pend.regulator.K = K


if __name__=='__main__':
    print("Proportional regulator:",reg.K)
    t0 = time.time()
    y = odeint(pend.x_dot_linear_regulator, x0, t)
    t1 = time.time() - t0
    print("Time of solving ode:",t1)

    fig1 = plt.figure()
    plt.plot(t, y[:, 2], label='angle')
    plt.plot(t, np.ones(t.shape) * np.pi,label='set point')
    plt.legend()
    fig1.show()

    fig2 = plt.figure()
    force = [reg.control(x,pend.set_point) for x in y]
    plt.plot(t, force, label='force')
    plt.legend()
    plt.show()



