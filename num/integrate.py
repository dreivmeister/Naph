import numpy as np
def simpson_rule(f, a, b, num):
    x = np.linspace(a,b,num)
    h = (b-a)/num
    
    I = f(a)+f(b)
    
    ls = 0
    for i in range(1,len(x)):
        ls += f(x[i])
    I += 2*ls
    rs = 0
    for i in range(len(x)-1):
        rs += f((x[i]+x[i+1])/2)
    I += 4*rs
    
    return (h/6)*I        

def gauß_quadrature(f, a, b, n):
    if n == 1:
        x = [0]
        alpha = [2]
    if n == 2:
        x = [-0.57735026919,0.57735026919]
        alpha = [1,1]
    if n == 3:
        x = [-0.774596669241,0,0.774596669241]
        alpha = [0.555555555556,0.888888888889,0.555555555556]
    # add n=4 and n=5
    
    I = 0
    for i in range(n):
        I += f((b-a)/2*x[i]+(a+b)/2)*alpha[i]
    I *= (b-a)/2
    
    return I


def romberg_integration(f,a,b,n,k):    
    def ai0(f, h, a, b):
        s = 0
        i = a + h
        while i <= (b - h):
            s += f(i)
            i += h
        s += (1 / 2) * f(a)
        s += (1 / 2) * f(b)
        return h * s

    def aik(h, a, i, k):
        m = (h[i - k] / h[i])**2
        a = m * a[i][k - 1] - a[i - 1][k - 1]
        return a / (m - 1)


    h = np.array([2**(-i) for i in range(n)])
    t = np.zeros((n, k))

    # erste spalte füllen
    for i in range(t.shape[0]):
        t[i, 0] = ai0(f, h[i], a, b)

    # rest füllen
    last_val = (0, 0)
    for k in range(1, t.shape[1]):
        for i in range(k, t.shape[0]):
            t[i, k] = aik(h, t, i, k)
            if t[i, k] != float(0):
                last_val = (i, k)

    return t[last_val]



if __name__=="__main__":
    def f(x):
        return x**2+4*x+2

    a=-1
    b=1
    # n = 100
    # print(gauß_quadrature(f, a, b, n=3))
    
    print(romberg_integration(f,a,b,5,5))
    
    