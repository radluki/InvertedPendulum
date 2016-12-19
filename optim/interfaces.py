from abc import abstractmethod
import numpy as np


class DifferentiableFunction(object):

    class GradientEvaluationType(object):
        ANALYTIC = 1
        SPSA = 2
        FD = 3

    def __init__(self, type_ = None):
        if type_ is None:
            self.set_gradient_evaluation(DifferentiableFunction.GradientEvaluationType.FD)
        else:
            self.set_gradient_evaluation(type_)

    @abstractmethod
    def valueAt(self,x):
        pass

    def set_gradient_evaluation(self,type):
        if type == DifferentiableFunction.GradientEvaluationType.SPSA:
            self.gradientAt = self.spsa
        elif type == DifferentiableFunction.GradientEvaluationType.FD:
            self.gradientAt = self.fd
        elif type == DifferentiableFunction.GradientEvaluationType.ANALYTIC:
            self.gradientAt = self.analytic_gradient
        else:
            raise Exception('Unknown GradientEvaluationType')

    @abstractmethod
    def analytic_gradient(self,x):
        pass

    def spsa(self,x):
        n = len(x)
        delta = 2*np.round(np.random.rand(n,1)) - 1
        eps = 1e-2
        xp = x + eps*delta
        xm = x - eps*delta
        ga = (self.valueAt(xp)-self.valueAt(xm))/(2*eps)/delta
        return ga

    def fd(self,x):
        n = len(x)
        eps = 1e-2
        g = np.zeros(n)
        def ei(i):
            e = np.zeros(n)
            e[i] = 1
            return e
        for i in range(n):
            g[i] = (self.valueAt(x+ei(i)*eps) - self.valueAt(x-ei(i)*eps))/(2*eps)

        return g

class InverseHessianApprox(object):

    def __init__(self, H0):
        self.H = H0

    @abstractmethod
    def update(self,s,y):
        """
        Updates hessian approximation H
        :param s: delta x
        :param y: delta gradient
        :return: successful update 0/1
        """
        pass

    def multiply(self, g):
        return self.H.dot(g)

    def reset(self):
        """Sets inverse hessian approximation to unit matrix"""
        self.H = np.eye(*self.H.shape)


class LineSearch(object):

    @abstractmethod
    def search(self,f,x0,d):
        """
        minimizes cost function on direction d
        :param f: cost function
        :param x0: initial point
        :param d: search direction
        :return: xp: minimal point on direction x0 + a*d
        """
        pass
