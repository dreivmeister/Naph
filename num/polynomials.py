import numpy as np
import math

def horners_method(poly, x):
    # evaluates the poly given as list of coeffs at x
    # poly = [a_n,a_n-1,...,a_1,a_0]
    
    poly_at_x = poly[0]
    for i in range(1, len(poly)):
        poly_at_x = poly_at_x * x + poly[i]
    return poly_at_x

#print(horners_method([1,0,0],2))


def extended_horners_method(poly, x):
    # evaluates the poly given as list of coeffs at x
    # and computes new poly with coeffs b of degree n-1
    # poly = [a_n,a_n-1,...,a_1,a_0]
    # x = x0
    # poly = (x - x0)*b + poly_at_x
    b = []
    b.append(poly[0])
    for i in range(1, len(poly)-1):
        b.append(b[i-1]*x+poly[i])
    poly_at_x = b[-1]*x + poly[-1]
    return poly_at_x, b

#print(extended_horners_method([1,0,0],2))


def tworow_horners_method(poly, q, p):
    # computes new poly with coeffs c of degree n-2 and A and B
    # poly = [a_n,a_n-1,...,a_1,a_0]
    # poly = (x**2 -px - q)*c + Ax + B
    # MAYBE LATER...
    pass


def lagrange_interpolation(x, y):
    pass


def newton_interpolation(x, y):
    # computes newton scheme results in interpolation polynomial in newton form
    n = len(x)
    d = np.zeros((n,n))
    for i in range(n):
        d[i,0] = y[i]
        
    for j in range(1, n):
        for i in range(n-j):
            d[i,j] = (d[i,j-1]-d[i+1,j-1])/(x[i]-x[i+j])
    return d[0,:]

print(newton_interpolation([0,1,2,3],[-1,0,5,20]))
    
def evaluate_newton_polynomial(c, x, x0):
    # evaluate polynomial given in newton form    
    y = c[-1]
    for j in range(len(x)-1, -1, -1):
        y = y*(x0 -x[j]) + c[j]
    return y
    
print(evaluate_newton_polynomial([-1,1,2,1],[0,1,2,3],1))


def hermite_interpolation():
    pass