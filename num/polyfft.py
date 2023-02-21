# multiplying polynomials in THETA(nlgn) using the FFT
import numpy as np
import time
import matplotlib.pyplot as plt
import math

def convert_to_point_value(f):
    return np.fft.rfft(f,n=2*len(f))

def convert_to_coefficient(f):
    return np.fft.irfft(f)


# f and g are in point value
# h (output) also in point value
def multiply_point_value(f, g):
    h = np.zeros(len(f),dtype = 'complex_')
    for i in range(len(f)):
        h[i] = f[i]*g[i]
    return h


def multiply_polynomials(f,g):
    return convert_to_coefficient(multiply_point_value(convert_to_point_value(f),
                                                       convert_to_point_value(g)))[:-1]


def multiply_coeff(f, g):
    res = [0]*(len(f)+len(g)-1)
    for o1,i1 in enumerate(f):
        for o2,i2 in enumerate(g):
            res[o1+o2] += i1*i2
    return res


def plot_results(m):
    fast_times = []
    for i in range(1,m):
        f = np.random.randint(-1000,1000,i)
        g = np.random.randint(-1000,1000,i)
        
        s = time.time()
        multiply_polynomials(f,g)
        e = time.time()
        fast_times.append(e-s)
        
        
    slow_times = []
    for i in range(1,m):
        f = np.random.randint(-1000,1000,i)
        g = np.random.randint(-1000,1000,i)
        
        s = time.time()
        multiply_coeff(f,g)
        e = time.time()
        slow_times.append(e-s)
        
    plt.plot(fast_times,label='fast')
    plt.plot(slow_times,label='slow')
    plt.legend()
    plt.show()


def expand_to_power(f):
    next = pow(2, math.ceil(math.log(len(f))/math.log(2)));
    for _ in range(next-len(f)):
        f.insert(0,0)
    return f


def fft(a):
    a = np.array(a)
    n = len(a)
    if n == 1:
        return a
    wn = np.exp((2*np.pi*1j)/n)
    w = 1
    a0 = a[0::2]
    a1 = a[1::2]
    y0 = fft(a0)
    y1 = fft(a1)
    y = np.zeros(n,dtype='complex_')
    for k in range(n//2):
        y[k] = y0[k] + w*y1[k]
        y[k+n//2] = y0[k] - w*y1[k]
        w = w*wn
    return y


if __name__=='__main__':
    f_coeff = [2,1,1,3,2]
    g_coeff = [7,1,2,9,5]
    
    print(convert_to_point_value(f_coeff))
    
    # expand_to_power(f_coeff)
    # expand_to_power(g_coeff)
    
    
    # print(multiply_polynomials(f_coeff,g_coeff))
    
    # h_coeff_s = multiply_coeff(f_coeff,g_coeff)
    # print(h_coeff_s)
