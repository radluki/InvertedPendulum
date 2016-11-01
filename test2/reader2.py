import time
from subprocess import call
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
from multiprocessing import Process


t = time.time()
reg = [1.0,2.04054585,-20.38379773,-3.93128902]
params = [0.5,0.2,0.1,0.3,0.006]
time_ = [0,10,100]
reg_s = [str(k) for k in reg]
params_s = [str(k) for k in params]
time_s = [str(k) for k in time_]
procs = []
n=30
for i in range(n):
    proc = subprocess.Popen(["./bt1"] +reg_s+params_s+time_s, stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    procs.append(proc)
arrs = []
for i in range(n):
    arr = np.reshape(np.fromstring(procs[i].stdout.read().decode("utf-8"),sep='\t'),(-1,5))
    arrs.append(arr)
print(len(arrs))
print(arrs[1][1,1])
t = time.time() - t
print("Time: ",t)
#plt.plot(arr[:,0],arr[:,3])
#plt.show()




