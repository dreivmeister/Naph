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


def f(x):
    return x**2+4*x+2

a=-1
b=1
n = 100


# print(np.linspace(a,b,n))


# print(f(-1))

# print(simpson_rule(f,a,b,n))




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

print(gauß_quadrature(f, a, b, n=3))
        