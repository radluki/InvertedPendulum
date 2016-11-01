import time
import sys
str1 = input("Waiting for input: ")
print("Received",str1)
i=0
while(i<2):
    i = i+1
    time.sleep(1)
    print("Iteration: ",i,file=sys.stdout)

str1 = input("Waiting for input.")
