import numpy as np
from math import exp, sin, cos

# simple forward mode AD with scalar variables
class DualNumber:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual # dual**2 = 0
        
    def __add__(self, x):
        if isinstance(x, DualNumber):
            return DualNumber(self.real+x.real,self.dual+x.dual)
        if isinstance(x, float):
            return DualNumber(self.real+x,self.dual)
    def __radd__(self, x):
        return self + x
    
    def __mul__(self,x):
        if isinstance(x, DualNumber):
            return DualNumber(self.real*x.real,self.real*x.dual+self.dual*x.real)
        if isinstance(x, float):
            return DualNumber(self.real*x,self.dual*x)
    def __rmul__(self, x):
        return self * x
    
# functions
def my_exp(x):
    if isinstance(x, DualNumber):
        return DualNumber(exp(x.real),exp(x.real)*x.dual)
def my_sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(sin(x.real),cos(x.real)*x.dual)
    
   

class DualArray:
    def __init__(self, real, dual):
        # for now assume real is a numpy array
        self.shape = real.shape
        self.real = real
        self.dual = dual # probably same shape as real
    
    # dont know if this is correct
    def __add__(self, x):
        if isinstance(x, DualArray):
            assert self.shape == x.shape
            return DualArray(self.real + x.real, self.dual + x.dual)

    def __mul__(self, x):
        # self @ x
        if isinstance(x, DualArray):
            assert self.shape[1] == x.shape[0]
            return DualArray(self.real @ x.real, self.dual @ x.real + self.real @ x.dual)
        


        
# only scalar case

# evaluate the function f at DualNumber(primal,tangent)
# most often the tangent = 1.0 (real number)
# and the primal = x (the real value of the input)
def pushforward(f, primal, tangent):
    if isinstance(primal, float):
        input = DualNumber(primal, tangent)
    if isinstance(primal, np.ndarray):
        input = DualArray(primal, tangent)
        
    output = f(input)
    primal_out = output.real
    tangent_out = output.dual
    return primal_out, tangent_out


def derivative(f, x):
    if isinstance(x, float):
        v = 1.0
    if isinstance(x, np.ndarray):
        v = np.ones_like(x) # creates dual array
        
    _, df_dx = pushforward(f, x, v)
    return df_dx

# computes the derivative of f at x
#print(pushforward(f,x_p))
#print(derivative(f,x_p))


# only vector case


# x_d = np.array([[1.0,1.0], 
#                 [1.0,1.0]])

# x = DualArray(x_r,x_d)



x_p = 3.0
def f(x):
    return my_sin(x)

x_v = np.array([[1.0,1.0], 
                [3.0,3.0]])
def fv(x):
    return x * x + x

if __name__=="__main__":
    # df_dx_v = derivative(f,x_p)
    # print(df_dx_v)
    
    # print(derivative(f,x_p))
    # print(cos(3.0))
    
    
    A = np.array([[1.0,2.0], 
                  [3.0,4.0]]) # 2x2
    b = np.array([0.0,1.0])
    print(b @ A)

    