import numpy as np
from math import e
import matplotlib.pyplot as plt

def euler_method(a,b,h,f,iv,plot=True):
    def step(y_n, t_n, h, f):
        return y_n + h*f(t_n,y_n)
    
    x = np.linspace(a,b,num=int((b-a)/h)+1)
    y = np.zeros(shape=x.shape)
    # inital value
    y[0] = iv
    
    # calc values
    for i in range(1,len(x)):
        y[i] = step(y[i-1],x[i-1],h,f)
        
    if plot:
        plt.plot(x,y,label='euler method')
        plt.scatter(x,y)
        plt.legend()
        plt.show()
    return x,y




if __name__=='__main__':
    a = 0
    b = 5
    h = 0.5
    def f(t,y):
        return np.sin(t)**2*y
    
    euler_method(a,b,h,f,2.0)