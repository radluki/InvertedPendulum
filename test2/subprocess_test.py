import subprocess
import time

with subprocess.Popen(["ifconfig"], stdout=subprocess.PIPE) as proc:
    print(proc.stdout.read().decode("utf-8"))

with subprocess.Popen(["python3", "/home/lukir/Documents/ipython/boost/test2/printing_iter.py"], stdout=subprocess.PIPE,stdin=subprocess.PIPE) as proc:
    print("Writing hello to pipe")
    proc.stdin.write("hello\n".encode("utf-8"))
    proc.stdin.flush()    

    with subprocess.Popen(["python3", "/home/lukir/Documents/ipython/boost/test2/printing_iter.py"], stdout=subprocess.PIPE,stdin=subprocess.PIPE) as proc2:
        print("Writing hello2 to pipe")
        proc2.stdin.write("hello2\n".encode("utf-8"))
        proc2.stdin.flush()
        time.sleep(2)
        print(proc2.stdout.read().decode("utf-8"))
    
    print(proc.stdout.read().decode("utf-8"))
