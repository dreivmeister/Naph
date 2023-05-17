import numpy as np


# this is the fundamental building block for the whole 
# autograd engine/deep learning framework
class Variable():
    __counter = 0
    def __init__(self, data, is_leaf=True, backw_func=None, requires_grad=True):
        # error handling
        if backw_func is None and not is_leaf:
            raise ValueError('non leaf nodes require a backward function')
        if np.isscalar(data):
            data = np.ones(1)*data
        if not isinstance(data,np.ndarray):
            raise ValueError(f'data should be of type "numpy.ndarray" or a scalar,but received {type(data)}')
        # set member variables
        self.data = data
        self.requires_grad = requires_grad
        
        if self.requires_grad:
            # assign class scope counter to this object
            self.id = Variable.__counter
            Variable.__counter += 1
            self.is_leaf = is_leaf
            self.prev = []
            self.backw_func = backw_func
            # set self.grad to zeros of shape self.data
            self._zero_grad()
            
            #TODO:
            #self.grad = Variable(np.zeros(data.shape),requires_grad=False)
    
        
    def backward(self):
        self.backw_func(self.grad)
    
    def _zero_grad(self):
        self.grad = np.zeros(self.data.shape,dtype=np.float64)
    
    def _step(self, learning_rate):
        # execute one step of gradient descent
        self.data = self.data - learning_rate*self.grad
    
    def __repr__(self):
        return f'Variable(id:{self.id},prev:{list(map(lambda a:a.id,self.prev))},is_leaf:{self.is_leaf})\n'
    
    def __add__(self, other):
        if not (isinstance(other,Variable)):
            raise ValueError('other needs to be a Variable')

            
        def backward_function(grad):
            self.grad = self.grad + grad
            other.grad = other.grad + grad
        
        res = Variable(self.data + other.data, is_leaf=False, backw_func=backward_function)
        res.prev.extend([self,other])
        return res
                
    
    def __sub__(self, other):
        if not (isinstance(other,Variable)):
            raise ValueError('other needs to be a Variable')
    
        def backward_function(dy):
            other.grad = other.grad - dy
            self.grad = self.grad + dy
            
        res = Variable(self.data - other.data, is_leaf=False, backw_func=backward_function)
        res.prev.extend([self,other])
        return res
    
    def __mul__(self, other):
        # elementwise multiply
        if not (isinstance(other,Variable)):
            raise ValueError('other needs to be a Variable')
    
        def backward_function(dy):
            if np.isscalar(dy):
                dy = np.ones(1)*dy
            self.grad = self.grad + np.multiply(dy, other.data)
            other.grad = other.grad + np.multiply(dy, self.data)
            
        res = Variable(self.data * other.data, is_leaf=False, backw_func=backward_function)
        res.prev.extend([self,other])
        return res
    
    def __pow__(self, power):
        if not (isinstance(power,int) or isinstance(power,float)):
            raise ValueError('power needs to be int or float')
        
        def backward_function(dy):
            self.grad = self.grad + (power*self.data**(power-1))*dy
            
        res = Variable(self.data**power, is_leaf=False, backw_func=backward_function)
        res.prev.append(self)
        return res
    
    def __truediv__ (self, other):
        if not (isinstance(other,Variable)):
            raise ValueError('other needs to be a Variable')
    
        def backward_function(dy):
            self.grad = self.grad + (1./other.data)*dy
            other.grad = other.grad + (-self.data/other.data**2)*dy
        
        res = Variable(self.data / other.data, is_leaf=False, backw_func=backward_function)
        res.prev.extend([self,other])
        return res
    
    def __neg__(self):
        return Variable(-1) * self

# define some primitive operations to link different Variable objects
# and to differentiate through these operations
def sum(a, ax=None):
    if not (isinstance(a,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy=1):
        if dy.ndim == 1:
            dy = np.expand_dims(dy,axis=1)
            if ax == 0:
                a.grad = a.grad + np.ones(a.data.shape)*dy.T 
            else:
                a.grad = a.grad + np.ones(a.data.shape)*dy
        else:
            a.grad = a.grad + np.ones(a.data.shape)*dy.T
        
        
    
    res = Variable(np.sum(a.data, axis=ax), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def transpose(a):
    if not (isinstance(a,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy=1):
        a.grad = a.grad + dy.T

    res = Variable(a.data.T, is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res
    

def dot(a,b):
    # dot product of two matrices
    # matrix multiply
    if not (isinstance(a,Variable) or isinstance(b,Variable)):
            raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad = a.grad + np.dot(dy, b.data.T)
        b.grad = b.grad + np.dot(a.data.T, dy)
    
    res = Variable(np.dot(a.data, b.data), is_leaf=False, backw_func=backward_function)
    res.prev.extend([a,b])
    return res


def relu(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy=1):
        a.grad[a.data>0] += dy[a.data>0]

    res = Variable(np.maximum(a.data, 0), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def tanh(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad += (1-np.tanh(a.data)**2)*dy
        
    res = Variable(np.tanh(a.data), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def sigmoid(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad = a.grad + (np.exp(-a.data)/(1+np.exp(-a.data)**2))*dy
    
    res = Variable(1/(1 + np.exp(-a.data)), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def max(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad += (np.where(a.data == np.max(a.data), 1., 0.))*dy

    res = Variable(np.max(a.data), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def mean(a, ax=None):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad += (1./(a.data.shape[0]*a.data.shape[1]))*dy

    res = Variable(np.mean(a.data, axis=ax), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def variance(a, ax=None):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    m = mean(a)
    n = a.data.shape[0]*a.data.shape[1]
    def backward_function(dy):
        a.grad += ((2./n)*(a.data - m.data))*dy
    
    res = Variable(np.var(a.data, axis=ax), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def exp(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad += np.exp(a.data)*dy
        
    res = Variable(np.exp(a.data), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def log(a):
    if not (isinstance(a,Variable)):
        raise ValueError('a needs to be a Variable object')
    
    def backward_function(dy):
        a.grad = a.grad + (1./(a.data+1e-12))*dy
        
    res = Variable(np.log(a.data), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res



# topological sort algorithm to sort the DAG which is the
# computational graph
def top_sort(var):
    visited = set()
    topo = []
    def top_sort_helper(v):
        if v in visited or v.is_leaf:
            pass
        else:
            visited.add(v)
            for child in v.prev:
                top_sort_helper(child)
            topo.append(v)
    top_sort_helper(var)
    return topo


def backward_graph(var):
    if not isinstance(var,Variable):
        raise ValueError('var needs to be a Variable instance')
    tsorted = top_sort(var)
    
    var.grad = np.ones(var.data.shape,dtype=np.float64)
    for var in reversed(tsorted):
        var.backward()
        