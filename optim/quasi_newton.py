from optim.functions import *
from  optim.interfaces import *
from optim.line_search_algorithms import *
import logging

logging.basicConfig(level=logging.WARNING)


class BfgsApprox(InverseHessianApprox):
    """
    straightforward implementation
    of BFGS update
    no code optimization involved
    """
    def __init__(self,H0):
        self.H = H0
        self.k = 0

    def update(self,s,y):
        self.k += 1
        if self.k == 2*len(s):
            self.k = 0
            self.H = np.eye(len(s))
        I = np.eye(*self.H.shape)
        ro_inv = np.inner(y,s) # ro = (y^T*s)^(-1)
        if ro_inv == 0:
            return 0
        ro = 1/ro_inv
        self.H = (I - ro*s*y.T).dot(self.H.dot(I - ro*y*s.T)) + ro*s*s.T
        return 1


class LBfgsApprox(InverseHessianApprox):

    def __init__(self,H,limit=-1):
        self.H = H
        self.limit = limit
        self.s = []
        self. y = []

    def update(self,s,y):
        if self.limit > 0 and len(self.s) >= self.limit:
            del self.s[0]
            del self.y[0]
        self.s.append(s)
        self.y.append(y)

    def multiply(self, g):
        r = g
        y = self.y
        s = self.s
        a = [None] * len(self.s)
        ro = [None] * len(self.s)
        # b = [None] * len(self.s)
        for i in reversed(range(len(self.s))):
            ro[i] = 1/(y[i].T.dot(s[i]))
            a[i] = ro[i]*s[i].T.dot(r)
            r = r - a[i]*y[i]

        r = self.H.dot(r)

        for i in range(len(s)):
            b = ro[i]*y[i].T.dot(r)
            r = r + (a[i] - b)*s[i]
        return r

    def reset(self,H=None):
        self.s = []
        self.y = []
        if not (H is None):
            self.H = H


def quasi_newton_solver(f, x0, \
                        inv_hessian_approx, \
                        tol=1e-5, max_iter=100, \
                        line_searcher=MinSearch()):
    """
    General Quasi Newton solver.
    :param f: cost function
    :param x0: starting point
    :param b0: initial hessian inverse
    :param tol: minimal square gradient length
    :param max_iter: maximal number of line searches
    :param inv_hessian_approx: object implementing QuasiNewtonApproximation
    :param line_searcher:
    :return: optimal solution, history, number of iterations, percent of successful updates
    """
    x = x0
    g = f.gradientAt(x)
    X = [x] # VIS
    update_counter = 0.0

    for n in range(max_iter):
        d = -inv_hessian_approx.multiply(g)
        # if d is not an improvement direction, then quasi newton is reset
        if d.T.dot(g) >= 0:
            inv_hessian_approx.reset()
            d = -g
        x_old = x
        x = line_searcher.search(f, x, d)
        X.append(x)
        g_old = g
        g = f.gradientAt(x)
        if g.T.dot(g) < tol:
            max_iter = n
            break
        s = x - x_old
        y = g - g_old
        update_counter += inv_hessian_approx.update(s, y)
    return x,np.array(X), max_iter, update_counter/max_iter*100


def press_to_end():
    try:
        input('Press to end: ')
    except:
        pass


def vector(*x):
    x = np.array(x)
    x = x.reshape(len(x))
    return x









