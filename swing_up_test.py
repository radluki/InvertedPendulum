import time
import matplotlib.pyplot as plt
import numpy as np
from pendulum import InvertedPendulum
from scipy.integrate import odeint
from scipy.optimize import  minimize
from optim.interfaces import DifferentiableFunction
from optim.line_search_algorithms import MinSearch
from optim.quasi_newton import BfgsApprox,quasi_newton_solver
from regulators import RegulatorLQR



class InvertedPendulumCost(DifferentiableFunction):

    def __init__(self, fun, type_ = None):
        super(InvertedPendulumCost,self).__init__(type_)
        self.fun = fun

    def valueAt(self, x):
        return self.fun(x)


def plot_state(t, y, pend):
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
    axes = (ax1, ax2, ax3, ax4, ax5)
    f.subplots_adjust(hspace=0)
    # angle visualization
    ax1.plot(t, 180/np.pi*y[:, 2], label=r'$\theta$')
    for i in range(int(np.floor(np.min(y[:,2]/np.pi))),int(np.ceil(np.max(y[:,2]/np.pi)))+1):
        ax1.plot(t, i*np.ones(t.shape) * 180)
    #ax1.plot(t, -np.ones(t.shape) * np.pi, label=r'$-\pi$')
    ax1.margins(1)
    ax1.legend()
    ax1.set_ylabel(r"$\theta$ $\mathrm{\left[rad\right]}$")
    ax1.grid(True)

    # unit = 1
    # y_tick = np.arange(-2, 2 + unit, unit)
    #
    # y_label = [r"$-2\pi$", r"$-\pi$", r"$0$", r"$+\pi$", r"$+2\pi$"]
    # ax1.set_yticks(y_tick * np.pi)
    # ax1.set_yticklabels(y_label)


    ax2.plot(t, y[:, 3], label=r'$\dot \theta$')
    ax2.plot(t, np.zeros(t.shape), label=r'$\dot \theta = 0$')
    ax2.margins(1)
    ax2.legend()
    ax2.set_ylabel(r"$\dot \theta$ $\mathrm{\left[\frac{rad}{s}\right]}$")
    ax2.grid(True)

    # Force visualization
    force = [pend.calculate_force(k) for k in y]
    ax3.plot(t, force, label=r'$F$')
    ax3.plot(t, np.zeros(t.shape), label=r'$F = 0$')

    en = [pend.lqr_cond2(k[2],k[3])>0 for k in y]
    beg = list()
    end = list()
    if en[0]>0:
        beg.append(0)
    for k in range(len(en)-1):
        if en[k+1]==1 and en[k]==0:
            beg.append(k+1)
        elif en[k+1]==0 and en[k]==1:
            end.append(k+1)
    if len(beg)>len(end):
        end.append(len(en)-1)

    for ax in axes:
        for k in range(len(beg)):
            ax.axvspan(t[beg[k]],t[end[k]],facecolor='#00FF00',alpha=0.2)#,

    #ax3.plot(t,en)

    ax3.set_ylim(np.min(force) * 1.2, np.max(force) * 1.2)
    ax3.margins(1)
    ax3.legend()
    ax3.set_ylabel(r"$F$ $\mathrm{[N]}$")
    ax3.grid(True)

    ax4.plot(t, y[:, 0], label=r'$x$')
    ax4.plot(t, np.zeros(t.shape), label=r'$x = 0$')
    ax4.margins(1)
    ax4.legend()
    ax4.set_ylabel(r"$x$ $\mathrm{[m]}$")
    ax4.grid(True)

    ax5.plot(t, y[:, 1], label=r'$\dot x$')
    ax5.plot(t, np.zeros(t.shape), label=r'$\dot x = 0$')
    ax5.margins(1)
    ax5.legend()
    ax5.set_ylabel(r"$\dot x$ $\mathrm{\left[\frac{m}{s}\right]}$")
    ax5.grid(True)
    plt.xticks(range(1,int(np.max(t))))

    plt.xlim(t[0], t[-1])
    plt.xlabel(r"$t$ $\mathrm{[s]}$")

    f.show()

def cost(x,Q,R,pend,t):
    pend.Fc = x[0]
    pend.x_swing = x[1]
    #pend.percent_energy = x[2]

    x0 = [0, 0, np.pi * 1e-5, 0]
    y = odeint(pend.x_dot_with_regulator, x0, t)
    eps = [pend.epsilon(k) for k in y]
    dt = t[1]-t[0]
    force = [pend.calculate_force(k) for k in y]
    cost = 0
    for ei in eps:
        cost += ei.dot(Q).dot(ei)
    for f in force:
        cost += f**2*R
    cost *= dt
    cost = cost[0,0]
    print('x:', x)
    print('cost:', cost)
    # cost = np.sqrt(np.linalg.norm(eps)**2+np.linalg.norm(force)**2)**2*dt
    # plt.plot(t,np.array(eps)**2)
    # plt.show()
    return cost


class RK4(object):

    def __init__(self,f,t0,x0,h):
        self.f = f
        self.t = t0
        self.x = x0
        self.h = h

    def integrate(self):
        tn = self.t
        xn = self.x
        h = self.h
        f = self.f
        k1 = f(tn, xn)
        k2 = f(tn + 0.5*h, xn + 0.5*h*k1)
        k3 = f(tn + 0.5*h, xn + 0.5*h*k2)
        k4 = f(tn + h, xn + h*k3)
        self.t += h
        self.x += h/6*(k1 + 2*k2 + 2*k3 + k4)
        return self.x


def fixed_step_cost(x,Q,R,pend,t):
    pend.Fc = x[0]
    pend.x_swing = x[1]

    x0 = [0, 0, -np.pi * 1e-5, 0]

    h = t[1]-t[0]
    y = list()
    yi = np.array(x0)
    y.append(yi)
    print("x = [",x[0],", ",x[1],"]")
    solver = RK4(pend.x_dot_with_regulator_ode,t[0],x0,h)
    for ti in t[1:]:
        yi = solver.integrate()
        #print(ti)
        y.append(yi.tolist())
    y = np.array(y)

    # eps = [pend.epsilon(k) for k in y]
    eps = [k - np.array([0,0,np.pi,0]) for k in y]

    dt = t[1] - t[0]
    force = [pend.calculate_force(k) for k in y]
    cost = 0
    for ei in eps:
        cost += ei.dot(Q).dot(ei)
    for f in force:
        cost += f ** 2 * R
    cost *= dt
    cost = cost[0, 0]
    print('cost:', cost)
    # cost = np.sqrt(np.linalg.norm(eps)**2+np.linalg.norm(force)**2)**2*dt
    # plt.plot(t,np.array(eps)**2)
    # plt.show()
    if np.isinf(np.any(y)) or np.isnan(np.any(y)):
        cost = np.inf
    return cost



if __name__=='__main__':
    # pendulum parameters
    M = 1
    m = 0.6
    b = 0.1
    I = 0.06
    l = 0.3
    # M = 0.548
    # m = 0.11
    # l = 0.1436
    # I = 0.003478
    # b = 0.001

    pend = InvertedPendulum(M, m, b, l, I)
    Q = np.eye(4)
    Q[2,2] *= 1e0
    Q[0,0] *= 1e0
    R = np.eye(1)*1e0
    reg = RegulatorLQR(pend.A, pend.B, Q=Q, R=R)
    pend.regulator = reg
    # pend.Fc = 4.97519462e+03
    # pend.x = -8.72676435e+06
    # pend.Fc = 11.68146648
    # pend.x_swing = 0.5
    # x0 = [0, 0, np.pi * 0.75, 10]
    # x0 = [0, 0, np.pi * 2.7, 0]
    # x0 = [0,0,np.pi*1e-5,4]
    x0 = [0, 0, np.pi * 1e-1, 0]
    t = np.linspace(0, 40, 1e3)




    print("Proportional regulator:",reg.K)
    t0 = time.time()
# ODEINT
    y = odeint(pend.x_dot_with_regulator, x0, t)

# ODE
#     solver = ode(pend.x_dot_with_regulator_ode)
#     solver.set_integrator('dopri5')
#     solver.set_initial_value(x0, t[0])
#     y = list()
#     for ti in t:
#         solver.integrate(ti)
#         print(ti)
#         y.append(solver.y.tolist())
#     y = np.array(y)


# Fixed step
#     h = t[1]-t[0]
#     y = list()
#     yi = np.array(x0)
#     y.append(yi)
#     solver = RK4(pend.x_dot_with_regulator_ode,t[0],x0,h)
#     for ti in t[1:]:
#         yi = solver.integrate()
#         print(ti)
#         y.append(yi.tolist())
#     y = np.array(y)
# END Fixed step

    t1 = time.time() - t0
    print("Time of solving ode:",t1)

    if 0:
        # angle visualization
        fig1 = plt.figure()
        plt.subplot(511)
        plt.plot(t, y[:, 2], label=r'$\theta$')
        plt.plot(t, np.ones(t.shape) * np.pi, label=r'$\pi$')
        plt.plot(t, -np.ones(t.shape) * np.pi, label=r'$-\pi$')
        plt.legend()
        #fig1.show()

        plt.subplot(512)
        plt.plot(t, y[:, 3], label=r'$\dot \theta$')
        plt.plot(t, np.zeros(t.shape), label=r'$\dot \theta = 0$')
        plt.legend()

        # Force visualization
        plt.subplot(515)
        force = [pend.calculate_force(k) for k in y]
        plt.plot(t, force, label=r'$F$')
        plt.plot(t, np.zeros(t.shape), label=r'$F = 0$')
        plt.ylim(np.min(force)*1.1,np.max(force)*1.1)
        plt.legend()

        plt.subplot(513)

        plt.plot(t, y[:,0], label=r'$x$')
        plt.plot(t, np.zeros(t.shape), label=r'$x = 0$')
        plt.legend()

        plt.subplot(514)

        plt.plot(t, y[:, 1], label=r'$\dot x$')
        plt.plot(t, np.zeros(t.shape), label=r'$\dot x = 0$')
        plt.legend()
        plt.show()
    elif 1:
        plot_state(t,y,pend)
        en = [pend.pendulum_energy(k[2],k[3]) - pend.m*pend.g*pend.l*(1-pend.percent_energy) for k in y]
        plt.figure(2)
        plt.plot(t,en)
        print("end state k*pi, k = ",y[-1,2]/np.pi)
        print("cost",cost([2,0.01,1],Q,R,pend,t))
        plt.show()
    elif 0:
        Q[2, 2] = 1e5
        def cost2(x):
            return fixed_step_cost(x, Q, R, pend, t)#+1e3*(np.min([0,x[0]]) + np.min([0,x[1]])) + 1e3*x[1]**2
        f10 = plt.figure(10)
        def callback(xk):
            print("////////////////////")
            plt.scatter(xk[0], xk[1])
            f10.show()
            time.sleep(2)
            print(xk)
            print("////////////////////")
        plt.figure()

        fun = InvertedPendulumCost(cost2,type_=DifferentiableFunction.GradientEvaluationType.FD)
        bfgs = BfgsApprox(np.eye(2))
        x0_bfgs = np.array([pend.Fc,pend.x_swing])
        line_searcher = MinSearch()
        ans = quasi_newton_solver(fun, x0_bfgs, bfgs, line_searcher=line_searcher)

    else:
        Q[2,2] = 1e1
        values = list()
        def cost2(x):
            values.append(fixed_step_cost(x,Q,R,pend,t))
            return values[-1]
        def callback(xk):
            print("////////////////////")
            print(xk)
            print("////////////////////")
        x0_bfgs = [pend.Fc,pend.x_swing]
        x0_bfgs = [ 1.9138111 ,  1.39984307]#[2, 1.5]
        #x0_bfgs = [ 18.8144922  ,  1.63521862]
        print("cost",cost2(x0_bfgs))
        # x = fmin_bfgs(cost2,x0_bfgs,callback=callback,gtol=1e-8,maxiter=1000)
        x = minimize(cost2, x0_bfgs, callback=callback,tol=1e-3)
        # x = basinhopping(cost2, x0_bfgs, callback=callback, stepsize=1)
        # x = x0_bfgs
        # x = [ 18.8144922  ,  1.63521862]
        print("Success:",x.success)
        print("Message",x.message)
        print(x)
        pend.Fc = x.x[0]
        pend.x_swing = x.x[1]
        y = odeint(pend.x_dot_with_regulator, x0, t)
        plot_state(t,y,pend)
        plt.figure()
        import pickle
        pickle.dump(values,open("values.pickle","w"))
        plt.plot(values)
        plt.show()





