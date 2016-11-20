import numpy as np
import matplotlib.pyplot as plt

def satur(t):
    return 0.5*(np.tanh(t)+1)
t = np.linspace(-10,10,1e3)
y1 = satur(t)
y2 = satur(1e1*t)

plt.plot(t,y1,label='1')
plt.plot(t,y2,label='2')
plt.legend()
plt.show()