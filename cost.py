import numpy as np
from abc import ABC


class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


class CostFunction(ABC):

    @abstractstatic
    def cost(e):
        pass



class RegulationTime(CostFunction):

    @staticmethod
    def cost(e):
        for i in reversed(range(len(e))):
            if np.abs(e[i,2]) > 0.01:
                #cost = sum(np.abs(e[:i,2]))
                return i#cost
        # if always too close
        return 1e10


class SumOfErrorSquared(CostFunction):

    @staticmethod
    def cost(e):
        err = sum(e[:,2]**2)
        return err


## For testing purposes
# class Test(CostFunction):
#
#     def hello(self):
#         print("hello")
#
#     @staticmethod
#     def cost(e):
#         print("cost fun")