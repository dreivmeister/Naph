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
        self.grad = np.zeros(self.data.shape)
    
    def _step(self, learning_rate):
        # execute one step of gradient descent
        self.data -= learning_rate*self.grad
    
    def __repr__(self):
        return f'Variable(id:{self.id},prev:{list(map(lambda a:a.id,self.prev))},is_leaf:{self.is_leaf})\n'
    

# define some primitive operations to link different Variable objects
# and to differentiate through these operations

# this primitive is commented more extensive than the others for demonstrational purposes
def plus(a, b):
    # error handling
    if not (isinstance(a,Variable) or isinstance(b,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    # dy is the upstream gradient in the computational graph
    # define how the gradient 'flows' through this operation
    def backward_function(dy):
        a.grad += dy
        b.grad = b.grad + dy
    
    # create new node in the computational graph as the result of the operation
    res = Variable(a.data + b.data, is_leaf=False, backw_func=backward_function)
    # add children information
    res.prev.extend([a,b])
    return res


def add(a,b):
    # elementwise vector sum
    # a is the output of dot(x,W) (a vector)
    # b is the bias vector
    
    if not (isinstance(a,Variable) or isinstance(b,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy):
        a.grad += dy
        b.grad += dy.sum(axis=0)
        
    res = Variable(np.add(a.data, b.data), is_leaf=False, backw_func=backward_function)
    res.prev.extend([a,b])
    return res

def minus(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
        
    def backward_function(dy):
        b.grad += -dy
        a.grad = a.grad + dy
        
    res = Variable(a.data - b.data, is_leaf=False, backw_func=backward_function)
    res.prev.extend([a,b])
    return res


def sum(a):
    if not (isinstance(a,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy=1):
        a.grad += np.ones(a.data.shape)*dy
    
    res = Variable(np.sum(a.data), is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res


def transpose(a):
    if not (isinstance(a,Variable)):
        raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy=1):
        a.grad += dy.T

    res = Variable(a.data.T, is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
    return res
    

def dot(a,b):
    if not (isinstance(a,Variable) or isinstance(b,Variable)):
            raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad += np.dot(dy, b.data.T)
        b.grad += np.dot(a.data.T, dy)
    
    res = Variable(np.dot(a.data, b.data), is_leaf=False, backw_func=backward_function)
    res.prev.extend([a,b])
    return res


def multiply(a,b):
    if not (isinstance(a,Variable) or isinstance(b,Variable)):
            raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
    
    def backward_function(dy):
        if np.isscalar(dy):
            dy = np.ones(1)*dy
        a.grad += np.multiply(dy, b.data)
        b.grad = b.grad + np.multiply(dy, a.data)
        
    res = Variable(np.multiply(a.data, b.data), is_leaf=False, backw_func=backward_function)
    res.prev.extend([a,b])
    return res


def matmul(a,b):
    if not (isinstance(a,Variable) and isinstance(b,Variable)):
            raise ValueError('all arguments need to be instances of the Variable class\
            at least one of them is not.')
            
    def backward_function(dy):
        if a.requires_grad:
            a.grad = plus(a.grad,matmul(dy,transpose(b)))
        if b.requires_grad:
            b.grad = plus(b.grad,matmul(transpose(a),dy))
            
    res = Variable(np.matmul(a.data,b.data),is_leaf=False,backward_fun=backward_function)
    res.prev.extend([a,b])
    return res


def const_multiply(a,c):
    if not (isinstance(a,Variable) or isinstance(c,(int, float))):
        raise ValueError('a needs to be a Variable object, c needs to be one of (int, float)')
    
    def backward_function(dy=1):
        a.grad += dy*c
        
    res = Variable(c * a.data, is_leaf=False, backw_func=backward_function)
    res.prev.append(a)
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
        a.grad += (1./(a.data+1e-10))*dy
        
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
    
    var.grad = np.ones(var.data.shape)
    for var in reversed(tsorted):
        var.backward()
        
        
