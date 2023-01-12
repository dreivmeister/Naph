import numpy as np
import matplotlib.pyplot as plt


def forward_difference(f, x0, h):
    return (f(x0+h)-f(x0))/h

def zentral_difference(f, x0, h):
    # maybe check for right choice of h
    return (f(x0+h)-f(x0-h))/(2*h)


def differentiate(f, a, b, num):
    h = (b-a)/num
    x = np.linspace(a,b,num)
    y = [forward_difference(f,i,h) for i in x]
    
    plt.plot(x,f(x))
    plt.plot(x,y)
    plt.show()
    
    return y


def richardson_extrapolation(f,x0,n,k):
    def ai0(f,x0,h):
        return (f(x0+h)-f(x0))/h
    def aik(h,a,i,k):
        m = h[i-k]/h[i]
        a = m*a[i][k-1]-a[i-1][k-1]
        return a/(m-1)
    
    h = np.array([2**(-i) for i in range(n)])
    a = np.zeros((n,k))
    
    
    # erste spalte füllen
    for i in range(a.shape[0]):
        a[i,0] = ai0(f,x0,h[i])

    # rest füllen
    last_val = (0,0)
    for k in range(1,a.shape[1]):
        for i in range(k,a.shape[0]):
            a[i,k] = aik(h,a,i,k)
            if a[i,k] != float(0):
                last_val = (i,k)
    return a[last_val]


if __name__=="__main__":
    def f(x):
        return x**2
    #y_p = differentiate(f, -1, 1, 1000000)
    print(richardson_extrapolation(f,2,5,5))