import numpy as np
from timeit import default_timer as timer
from timeit import time as time

#1.1
def checktick():
    # Code taken from assignment page on Canvas
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 =  time.time_ns() # get timestamp from timer
        t2 = time.time_ns() # get timestamp from timer
        while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
            t2 = time.time_ns() # get timestamp from timer
        t1 = t2 # this is outside the loop
        timesfound[i] = t1 # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound) # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta


# time.time() and time.time_ns() has differs between runs: get average of n runs
n = 10
check_tick = 0
for i in range(n):
    # We have to change the method we run within the checktick() function at the 'get timestamp from timer' comments
    check_tick += checktick()

check_tick = check_tick / n

print(checktick())