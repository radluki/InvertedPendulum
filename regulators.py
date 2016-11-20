import numpy as np
from scipy.linalg import solve_continuous_are


class LinearRegulator(object):

    def __init__(self,K):
        self.K = K

    def control(self,x,set_point):
        dx = [a-b for a,b in zip(x,set_point)]
        return self.K.dot(dx)

class RegulatorLQR(LinearRegulator):

    def __init__(self,A,B,Q=np.eye(4),R=np.eye(1)):
        X = solve_continuous_are(A, B, Q, R)
        K = -np.linalg.inv(R)*B.T.dot(X)
        self.K = K
