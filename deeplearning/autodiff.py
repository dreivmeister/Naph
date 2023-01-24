import numpy as np
from math import exp, sin, cos, pi


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
    
    def __sub__(self, x):
        return self + (-1*x)
    def __rsub__(self, x):
        return self + (-1*x)
    
    def __mul__(self,x):
        if isinstance(x, DualNumber):
            return DualNumber(self.real*x.real,self.real*x.dual+self.dual*x.real)
        if isinstance(x, float):
            return DualNumber(self.real*x,self.dual*x)
    def __rmul__(self, x):
        return self * x
    
    def __pow__(self,n):
        if isinstance(n, int):
            return DualNumber(self.real**n,n*self.real**(n-1)*self.dual)
    
# scalar functions
def exp_(x):
    if isinstance(x, DualNumber):
        return DualNumber(exp(x.real),exp(x.real)*x.dual)
def sin_(x):
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

# evaluate the function f at DualNumber(primal,tangent)
# most often the tangent = 1.0 (real number)
# and the primal = x (the real value of the input)
def pushforward(f, primal, tangent):
    if isinstance(primal, float):
        input = DualNumber(primal, tangent)
    if isinstance(primal, np.ndarray):
        input = DualArray(primal, tangent)
    output = f(input)
    print(output)
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


def compute_jacobian(f, p):
    jacobian = np.zeros(shape=(len(f),len(p)))
    # p is a tuple
    # f is a list of component functions
    # go every function in f:
    # which is every row of the jacobian
    for i,comp_func in enumerate(f):
        # go over every dimension in p
        # for current comp_func
        for j in range(len(p)):
            # compute partial derivative of comp_func
            # with respect to current dimension
            
            # get those params right
            a = [*p[:j],*p[j+1:]]
            a.insert(j,None)
            def cf(p):
                a[j] = p
                return comp_func(*a)
            print(a)
            print(p[j])
            jacobian[i][j] = derivative(cf,p[j])
    return jacobian
        
    



if __name__=="__main__":
    # p = [1.0,2.1,3.5]
    # def f1(x,y,z):  
    #     return x*x + 3.0*y + z*z
    # def f2(x,y,z):
    #     return 10.0*x + y*z
    # def f3(x,y,z):
    #     return z*z + x + y
    # f = [f3,f2,f1]
    
    
    # print(compute_jacobian(f,p))
    
    
    
    # x_p = 2.0
    # y_p = 4.0
    # def f0(x):
    #     return x**4
    # def f(x,y):
    #     return x*x + x*y
    # def f1(x):
    #     return f(x,y_p)
    # def f2(y):
    #     return f(x_p,y)
    
    # #print(f2(y_p))
    # print(derivative(f0,x_p))
    

    x_p = 10.0
    def f(x):
        return 1.0 - (4.0*x)


    print(derivative(f,x_p))

    # x_v = np.array([[1.0,1.0], 
    #                 [3.0,3.0]])
    # def fv(x):
    #     return x * x + x
    