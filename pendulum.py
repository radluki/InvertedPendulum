import numpy as np


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
        self.force = list()
        self.a = 1e4
        self.Fc = 3.0 #3.8
        self.x_swing = 0.5
        self.percent_energy = 1e-2
        self.lqr_cond_rej = 0 # 0 swingup

    def x_dot(self,x,t=0,F=0):
        # x0 = x[0]
        x1 = x[1]
        theta0 = x[2]
        theta1 = x[3]
        # if not (np.linalg.det(self.mat1 + self.mat2*np.cos(theta0))**2 > 1e-6):
        #     print(x)
        #assert np.linalg.det(self.mat1 + self.mat2*np.cos(theta0))**2 > 1e-6
        x2,theta2 = np.linalg.inv(self.mat1 + self.mat2*np.cos(theta0)).dot(\
            [-self.b*x1 + self.m*self.l*theta1**2*np.sin(theta0) + F,\
             -self.m*self.g*self.l*np.sin(theta0)])

        x_prime = [x1,x2,theta1,theta2]
        # if np.any(np.isnan(x_prime)):
        #     print(x_prime)
        #     print("x",x)
        #     print("det",np.linalg.det(self.mat1 + self.mat2*np.cos(theta0)))
        #assert not np.any(np.isnan(x_prime))
        return x_prime

    def x_dot_with_regulator(self, x, t):

        F = self.calculate_force(x)
        x_dot = self.x_dot(x,t,F)

        return x_dot

    def x_dot_with_regulator_ode(self,t,x):
        """Interface for ode solver"""
        xdot = np.array(self.x_dot_with_regulator(x,t))
        return xdot

    def epsilon(self,x):
        """Calculates angle from upward position"""
        eps = x - self.set_point * self.stairs(x[2], 2 * np.pi) / np.pi
        return eps

    def calculate_force(self,x):
        set_point2 = self.set_point * self.stairs(x[2],2*np.pi)/np.pi   # enables multiple rotations
        F =  self.stabilizing_regulator(30,x,set_point2)# lqr
        F += self.swing_up_regulator(40,x,set_point2)

        return F#np.clip(F,-2*self.Fc,2*self.Fc)

    def stabilizing_regulator(self,zone_in_degrees,x,set_point2):
        in_zone = self.step(zone_in_degrees*np.pi/180 - np.abs(x[2]-set_point2[2]))
        #lqr_cond_val = self.lqr_cond(self.pendulum_energy(x[2],x[3]))
        return self.regulator.control(x, set_point2) * in_zone #* lqr_cond_val

    def swing_up_regulator(self,zone_in_degrees,x,set_point2):
        e = self.pendulum_energy(x[2], x[3])  # energy
        lqr_cond = self.lqr_cond(e)  # check energy

        swingup_direction = self.sign(-np.cos(x[2]) * x[3])
        increase_energy_cond = self.sign(self.m * self.g * self.l - e)
        change_energy_cond = (1 - lqr_cond)
        in_zone = self.step( -(zone_in_degrees * np.pi / 180 - np.abs(x[2] - set_point2[2])))
        # F = self.Fc * (self.x_swing * swingup_direction - x[0]) * increase_energy_cond*change_energy_cond*in_zone
        F = self.Fc * swingup_direction  * increase_energy_cond * change_energy_cond * in_zone

        return F

    def lqr_cond(self,e):
        """Required energy must be between bounds for lqr"""
        Emin = self.m * self.g * self.l * (1 - self.percent_energy)
        Emax = self.m * self.g * self.l * (1 + self.percent_energy)
        lqr_cond = self.square(e - Emin, Emax - e)
        return lqr_cond

    def lqr_cond2(self,x2,x3):
        """For visualization purposes"""
        set_point2 = self.set_point * self.stairs(x2, 2 * np.pi) / np.pi
        self.lqr_cond_rej = self.step(30 * np.pi / 180 - np.abs(x2 - set_point2[2]))
        return self.lqr_cond_rej

    def step(self, t):
        return 0.5 * (np.tanh(self.a*t) + 1)

    def sign(self, t):
        return np.tanh(self.a*t)

    def square(self,cond1,cond2):
        return self.step(cond1)*self.step(cond2)

    def stairs(self, t,step):
        val = -10.5 * step
        for i in range(-10,11):
            val += step * self.step(t-i*step)
        return val

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

    def pendulum_energy(self,angle,angle_dot):
        E = 0.5*self.I*angle_dot**2 - self.m*self.g*self.l*np.cos(angle)
        return E


import matplotlib.pyplot as plt

if __name__=="__main__":
    ip = InvertedPendulum(1,2,3,4,5)
    t = np.linspace(-12,12,1e5)
    plt.plot(t,ip.stairs(t,np.pi*2))
    plt.plot(t,t)
    plt.grid(True)
    plt.show()