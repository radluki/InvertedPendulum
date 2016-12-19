from optim.interfaces import *


class RosenbrockFunction(DifferentiableFunction):

    def __init__(self, a=1, b=100, type_ = None):
        super(RosenbrockFunction, self).__init__(type_)
        self.a = a
        self.b = b

    def valueAt(self ,x):
        return self._valueAt(x[0] ,x[1])

    def analytic_gradient(self,x):
        return self._gradientAt(x[0] ,x[1])

    def _valueAt(self ,x ,y):
        """ Calculates Rosenbrock function"""
        return (self. a -x ) **2 + self. b *(y - x** 2) ** 2

    def _gradientAt(self, x, y):
        """Calculates Rosenbrock function derivativ"""
        return np.array([[-2 * (self.a - x) + 2 * self.b * (y - x ** 2) * (-2 * x)], [2 * self.b * (y - x ** 2)]])


class Paraboloid(DifferentiableFunction):

    def __init__(self, a=10, b=1, type_ = None):
        super(Paraboloid,self).__init__(type_)
        self.a = a
        self.b = b

    def valueAt(self, x):
        return self._valueAt(x[0][0], x[1][0])

    def analytic_gradient(self,x):
        return self._gradientAt(x[0][0], x[1][0])

    def _valueAt(self, x, y):
        return self.a * x ** 2 + self.b * y ** 2

    def _gradientAt(self, x, y):
        return np.array([[2 * self.a * x], [2 * self.b * y]])


class Function4d(DifferentiableFunction):

    def __init__(self, a=4, b=1, c=1, d=100, type_ = None):
        self.para = Paraboloid(a, b, type_=type_)
        self.rosen = RosenbrockFunction(c, d, type_=type_)

    def valueAt(self,x):
        return self.rosen.valueAt(x[0:2]) + self.rosen.valueAt(x[2:4])

    def analytic_gradient(self,x):
        g1 = self.rosen.gradientAt(x[0:2]).tolist()
        g2 = self.rosen.gradientAt(x[2:4]).tolist()
        g = g1+g2
        return np.array(g)


class ParaboloidNd(DifferentiableFunction):

    def __init__(self, n, type_):
        super(ParaboloidNd, self).__init__(type_=type_)
        self.A = np.random.rand(n,n)
        self.A = np.inner(self.A, self.A)
        #self.A = A

    def valueAt(self,x):
        if len(x.shape) == 1:
            return 0.5*x.T.dot(self.A.dot(x))
        else:
            return 0.5*sum(x*np.tensordot(self.A,x,[[1],[0]]),0).ravel()

    def analytic_gradient(self,x):
        return np.tensordot(self.A,x,axes = [[1],[0]])