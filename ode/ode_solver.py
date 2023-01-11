import numpy as np
import matplotlib.pyplot as plt

def euler_method(a,b,h,f,iv,plot=True):
    def step(y_n, t_n, h, f):
        return y_n + h*f(t_n,y_n)
    
    t = np.linspace(a,b,num=int((b-a)/h)+1)
    y = np.zeros(shape=t.shape)
    # inital value
    y[0] = iv
    
    # calc values
    for i in range(1,len(t)):
        y[i] = step(y[i-1],t[i-1],h,f)
        
    if plot:
        plt.plot(t,y,label='euler method')
        plt.scatter(t,y)
        plt.legend()
        plt.show()
    return t,y

def runge_kutta_4(a,b,h,f,iv,plot=True):
    def step(y_n, k1, k2, k3, k4, h):
        return y_n + (1/6)*(k1+2*k2+2*k3+k4)*h
    
    t = np.linspace(a,b,num=int((b-a)/h)+1)
    y = np.zeros(shape=t.shape)
    # inital value
    y[0] = iv
    
    # calc values
    for i in range(1,len(t)):
        k1 = f(t[i-1],y[i-1])
        k2 = f(t[i-1]+h/2,y[i-1]+h*(k1/2))
        k3 = f(t[i-1]+h/2,y[i-1]+h*(k2/2))
        k4 = f(t[i-1]+h,y[i-1]+h*k3)
        y[i] = step(y[i-1],k1,k2,k3,k4,h)
        
    if plot:
        plt.plot(t,y,label='runge kutta 4')
        plt.scatter(t,y)
        plt.legend()
        plt.show()
    return t,y






if __name__=='__main__':
    a = 0
    b = 5
    h = 0.5
    def f(t,y):
        return np.sin(t)**2*y
    
    x,y1 = euler_method(a,b,h,f,2.0,plot=False)
    _,y2 = runge_kutta_4(a,b,h,f,2.0,plot=False)
    
    plt.plot(x,y1,label='euler')
    plt.plot(x,y2,label='rk4')
    plt.legend()
    plt.show()