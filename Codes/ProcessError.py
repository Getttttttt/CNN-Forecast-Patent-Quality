import pandas

from pyJoules.energy_meter import measure_energy
import time

@measure_energy
def foo():
    for i in range(10):
        print(i)
        time.sleep(2)
        t = i*3
        t -= i*1

foo()