import time
from typing import final

from numpy.lib.function_base import average
import zombie 
import numpy

noz=1
norb=100

results=numpy.zeros((noz,2))

for i in range(noz):
    states=[]
    #for j in range(2):
    #states.append(zombie.zom(norb,typ='aufbau',nel=6))
    states.append(zombie.zom(norb,typ='ran'))

    zs1=states[0].zs
    zs2=states[0].zs

    start=time.perf_counter()
    print(zombie.Stot1(zs1,zs2,norb))
    finish=time.perf_counter()

    results[i,0]=finish-start

    start=time.perf_counter()
    print(zombie.Stotfast(zs1,zs2,norb))
    #zombie.spsm2(zs1,zs2,norb)
    finish=time.perf_counter()

    results[i,1]=finish-start

    # start=time.perf_counter()
    # print(zombie.spsmfast(zs1,zs2,norb))
    # finish=time.perf_counter()

    # results[i,2]=finish-start

print(results)

avr=0
for i in range(noz):
    speed=results[i,0]/results[i,1]
    avr=avr+speed

fin=avr/noz
print(fin)
