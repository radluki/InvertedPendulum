from optim.functions import *
from optim.interfaces import *


class BacktrackingLineSearch(LineSearch):
    """
    From wikipedia, commonly used
    Stopping Condition:
    f(x + a*d) < f(x) + g*d*a*c
    c in (0,1), m=g*d is a directional derivative of f in direction d
    """
    def __init__(self,alpha = 1,c=0.7,tau=0.1,max_iter=50):
        """
        Stopping condition: f(x + alpha*d) < f(x) + g*d*alpha*c
        alpha update if condition is false: alpha = alpha*tau
        :param c: value in (0,1)
        :param tau: value in (0,1)
        :param max_iter: iterations limit
        """
        self.alpha = alpha
        self.c = c
        self.tau = tau
        self.max_iter = max_iter

    def search(self,f,x0,d):
        """ Finds first x that satisfies Armijo condition"""
        g = f.gradientAt(x0)
        m = g.T.dot(d)
        alpha = self.alpha

        for i in range(self.max_iter):
            x = x0 + alpha*d
            #TODO what if condition is not fulfilled
            if f.valueAt(x) - f.valueAt(x0) < self.c*m*alpha:
                break
            else:
                alpha *= self.tau
        return x


class MinSearch(LineSearch):

    class GoldenRatio(object):
        def __init__(self):
            self.max_iter = 5
            self.ratio = 0.5*np.sqrt(5) - 0.5

    class QubicApprox(object):
        def __init__(self):
            self.max_iter = 1
            self.epsilon = 1e-6

    class Contraction(object):
        def __init__(self):
            self.beta = 0.5
            self.epsilon = 1e-8

    class Expansion(object):
        def __init__(self):
            self.alpha = 3
            self.max_iter = 10

    def __init__(self):
        # common params
        self.interval = None # alphas which form interval containing min
        self.values = None # function values for interval
        self.interval_found = None # flag
        self.value_found = None # flag

        # states
        self.gr = self.GoldenRatio()
        self.qa = self.QubicApprox()
        self.contr = self.Contraction()
        self.expan = self.Expansion()

    def search(self,f,x0,d):
        """Uses expansion, golden ratio and qubic approximation"""
        self.find_interval(f,x0,d)
        if self.interval_found:
            self.golden_ratio_method(f,x0,d)
            self.qubic_approx(f,x0,d)
        if self.value_found:
            return x0 + self.interval[1]*d
        else:
            return x0

    def reset(self):
        self.interval = np.nan*np.eye(3,1).ravel()
        self.values = np.nan*np.eye(3,1).ravel()
        self.value_found = False
        self.interval_found = False

    def find_interval(self,f,x0,d):
        """ Expansion - contraction algorithm """

        self.reset()
        x = x0 + d
        value_x0 = f.valueAt(x0)
        value_x = f.valueAt(x)
        self.interval[0] = 0
        self.values[0] = value_x0

        if value_x0 > value_x:
            self.value_found = True
            self.interval[1] = 1
            self.values[1] = value_x
            self.expansion(f,x0,d)
        else:
            self.interval[2] = 1
            self.values[2] = value_x
            self.contraction(f,x0,d)

    def expansion(self,f,x0,d):
        alpha = 1
        for i in range(self.expan.max_iter):
            alpha *= self.expan.alpha
            x = x0 + alpha*d
            value = f.valueAt(x)
            if value > self.values[1]:
                self.interval_found = True
                self.interval[2] = alpha
                self.values[2] = value
                break
            else:
                self.interval[1] = alpha
                self.values[1] = value

    def contraction(self,f,x0,d):
        beta = 1
        while beta > self.contr.epsilon:
            beta *= self.contr.beta
            x = x0 + beta * d
            value = f.valueAt(x)
            if value < self.values[0]:
                self.value_found = True
                self.interval_found = True
                self.interval[1] = beta
                self.values[1] = value
                break
            else:
                self.interval[2] = beta
                self.values[2] = value

    def golden_ratio_method(self,f,x0,d):
        tau = self.gr.ratio
        for i in range(self.gr.max_iter):
            if not np.all(sorted(self.interval) == self.interval):
                print('Error in golden ratio: ' + str(self.interval))
            distances = self.interval - self.interval[1]
            ind = 2 if -distances[0] < distances[2] else 0
            x = x0 + self.interval[ind]*d - tau*distances[ind]*d
            value = f.valueAt(x)
            alpha = (x[0] - x0[0]) / d[0]
            self.__substitute(ind,alpha,value)

    def __substitute(self,ind,alpha,value):
        """
        Inserts new point to interval and values
        so that the interval is still sorted,
        and the values[1] is a minimal value
        """
        if value < self.values[1]:
            ind = -(ind - 2)  # change 2 -> 0, 0 -> 2
            self.interval[ind] = self.interval[1]
            self.values[ind] = self.values[1]
            self.interval[1] = alpha
            self.values[1] = value
        else:
            self.interval[ind] = alpha
            self.values[ind] = value

    def qubic_approx(self,f,x0,d):
        """
        Straightforward implementation
        probably much slower than optimal one
        """
        for i in range(self.qa.max_iter):
            Qd = self.values[0]
            Qw = self.values[1]
            Qg = self.values[2]
            zd = self.interval[0]
            zw = self.interval[1]
            zg = self.interval[2]

            if (zg - zd)/zw < self.qa.epsilon:
                break

            alpha_min = 0.5*(Qd*(zw**2 - zg**2) + Qw*(zg**2 - zd**2) + Qg*(zd**2 - zw**2))
            den = (Qd*(zw-zg) + Qw*(zg-zd) + Qg*(zd-zw))
            if den < 1e-8:
                return
            alpha_min /= (Qd*(zw-zg) + Qw*(zg-zd) + Qg*(zd-zw))
            x = x0 + alpha_min*d
            value = f.valueAt(x)
            ind = 0 if alpha_min < zw else 2
            assert ( self.interval[0] < alpha_min and self.interval[2] > alpha_min )
            self.__substitute(ind,alpha_min,value)
