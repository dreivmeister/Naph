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

x_p = 3.0

def f(x):
    return x * x



# evaluate the function f at DualNumber(primal,tangent)
# most often the tangent = 1.0 (real number)
# and the primal = x (the real value of the input)
def pushforward(f, primal, tangent=1.0):
    input = DualNumber(primal, tangent)
    output = f(input)
    primal_out = output.real
    tangent_out = output.dual
    return primal_out, tangent_out

# computes the derivative of f at x
def derivative(f, x): # x is a real number
    v = 1.0
    _, df_dx = pushforward(f, x, v)
    return df_dx



print(pushforward(f,x_p))
print(derivative(f,x_p))