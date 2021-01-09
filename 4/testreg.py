import numpy as np
from random import *
import matplotlib.pyplot as plt
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func1(x, a, b, c,d):
    return a*x**3+b*x**2+c*x+d

def func2(x, k,b):
    return k*x+b

print(func1(1,2.2,0.01,-4,5))
f = open("data.txt", "w")
f.write("0,1\n")
for i in range(50):
    x=uniform(-5,5)
    y=func(x,-5,0.45,2.2)+uniform(-3,3)
    s=str(x)+","+str(y)
    plt.scatter(x, y)
    f.write("%s\n" % s)
f.close()
plt.show()