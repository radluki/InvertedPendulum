import time
from subprocess import call
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


t = time.time()
reg = [1.0,2.04054585,-20.38379773,-3.93128902]
params = [0.5,0.2,0.1,0.3,0.006]
time_ = [0,10,100]
reg_s = [str(k) for k in reg]
params_s = [str(k) for k in params]
time_s = [str(k) for k in time_]
call(["./bt1"]+reg_s+params_s+time_s)
t = time.time() - t
print("Time: ",t)
#f = open("test3.txt")
#li = list(f)
#print(' '.join(li))
#df = pd.read_csv("test3.txt",delimiter="\t")
t = time.time()
# 0.000222 s when you specify time, else 2 times worse
arr = np.fromfile("test3.txt",sep='\t',count=-1)
#print(arr)

t = time.time() - t
print("Time: ",t)


