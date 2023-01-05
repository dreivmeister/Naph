import numpy as np
import matplotlib.pyplot as plt

# import sys

# if __name__ == "__main__":
#     print(f"Arguments count: {len(sys.argv)}")
#     for i, arg in enumerate(sys.argv):
#         print(f"Argument {i:>6}: {arg}")



def forward_difference(f, x0, h):
    return (f(x0+h)-f(x0))/h

def zentral_difference(f, x0, h):
    # maybe check for right choice of h
    return (f(x0+h)-f(x0-h))/(2*h)


def f(x):
    return x**2



def differentiate(f, a, b, num):
    h = (b-a)/num
    x = np.linspace(a,b,num)
    y = [forward_difference(f,i,h) for i in x]
    
    plt.plot(x,f(x))
    plt.plot(x,y)
    plt.show()
    
    return y


y_p = differentiate(f, -1, 1, 1000000)