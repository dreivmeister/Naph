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

def euler_step(y_n, h, f_val):
        return y_n + h*f_val

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

def runge_kutta_4_step(y_n,k1,k2,k3,k4,h):
    return y_n + (1/6)*(k1+2*k2+2*k3+k4)*h




if __name__=='__main__':
    # a = 0
    # b = 5
    # h = 0.5
    # def f(t,y):
    #     return np.sin(t)**2*y
    
    # x,y1 = euler_method(a,b,h,f,2.0,plot=False)
    # _,y2 = runge_kutta_4(a,b,h,f,2.0,plot=False)
    
    # plt.plot(x,y1,label='euler')
    # plt.plot(x,y2,label='rk4')
    # plt.legend()
    # plt.show()
    
    
    
    
    
    def f1(y,x,a=10):
        return a*(y-x)
    def f2(y,x,z,b=28):
        return x*(b-z)-y
    def f3(y,x,z,c=8/3):
        return x*y-c*z
    
    
    
    # init params
    a = 10
    b = 28
    c = 8/3
    # init other params
    t0 = 0
    t1 = 75
    h = 0.01
    # init data structures
    t = np.linspace(t0,t1,num=int((t1-t0)/h)+1)
    x = np.zeros(shape=t.shape)
    y = np.zeros(shape=t.shape)
    z = np.zeros(shape=t.shape)
    
    # init IVP
    x[0] = 0.
    y[0] = 1.
    z[0] = 1.05
    
    # step through calculation
    for i in range(1,len(t)):
        f1_val = f1(y[i-1],x[i-1])
        x[i] = euler_step(x[i-1],h,f1_val)
        
        f2_val = f2(y[i-1],x[i-1],z[i-1])
        y[i] = euler_step(y[i-1],h,f2_val)
        
        f3_val = f3(y[i-1],x[i-1],z[i-1])
        z[i] = euler_step(z[i-1],h,f3_val)
    
    
    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x,y,z)
    plt.show()
    
    
    
        
    