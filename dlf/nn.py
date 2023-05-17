import numpy as np
import engine
from engine import Variable


# the module parent class together with different implementations of it
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p._zero_grad()
            
    def step(self, lr):
        for p in self.parameters():
            if p != []:
                p._step(lr)

    def parameters(self):
        return []


class LinearLayer(Module):
    def __init__(self, features_in, features_out, w=None, b=None):
        super(LinearLayer, self).__init__()
        std = 1.0/features_in
        if w is not None:
            self.w = w
        else:
            self.w = Variable(np.random.uniform(-std, std, (features_in, features_out)))
            
        if b is not None:
            self.b = b
        else:
            self.b = Variable(np.random.uniform(-std, std, features_out))

    def forward(self, x):
        return engine.dot(x, self.w) + self.b
    
    def parameters(self):
        return [self.w, self.b]


class Conv1D(Module):
    def __init__(self) -> None:
        super().__init__()

class BatchNorm1D(Module):
    def __init__(self, num_features) -> None:
        super(BatchNorm1D, self).__init__()
        self.gamma = Variable(np.ones(num_features))
        self.beta = Variable(np.zeros(num_features))
        
    def forward(self, x):
        # mean
        m = engine.mean(x, ax=0)
        # variance
        v = engine.variance(x, ax=0)
        # normalize
        x = (x - m)/(v**(1/2))
        # scale and shift
        return self.gamma*x + self.beta
    
    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Module):
    def __init__(self, p_drop=0.5):
        super(Dropout, self).__init__()
        self.p_keep = 1-p_drop
        
    def forward(self, x):
        binary_value = Variable(np.random.rand(x.data.shape[0], x.data.shape[1]) < self.p_keep)
        res = x * binary_value
        res = engine.div(res,Variable(self.p_keep)) # inverted dropout
        return res
    
    def parameters(self):
        return super().parameters()
    
    



# misc functions and classes like:
# losses, optimizers, convencience utilities and so on
# regression loss
def mean_squared_error(out, target):
    # shapes have to match
    
    # get the batch dimension
    n = out.data.shape[1]*out.data.shape[0]
    r = out - target
    return Variable(1/n) * engine.sum(r*r)
    
    
# multi class classification
def cross_entropy(out, targets, eps=1e-12):
    """
    expected:
    predictions = np.array([[0.25,0.25,0.25,0.25],
                            [0.01,0.01,0.01,0.96]])
    targets = np.array([[0,0,0,1],
                        [0,0,0,1]]) (2,4)
    # 2 - num samples
    # 4 - num classes
    # with one hot vectors
    for binary classification:
    - softmax on two neuron output
    predictions = np.array([[0.5,0.5],
                            [0,04,0.96]])
    targets = np.array([[1,0],
                        [0,1]]) (2,2)
                        
    """
    #out.data = np.clip(out.data, eps, 1. - eps)
    N = out.data.shape[0]
    ce = engine.sum(targets * engine.log(out+Variable(eps)))
    print('ce',ce.data.shape)
    ce = Variable(-(1/N)) * ce
    return ce
    
    
def softmax(out):
    out_exp = engine.exp(out - engine.max(out))
    exp_sum = engine.sum(out_exp, ax=1)
    exp_sum = Variable(1.) / exp_sum
    return engine.transpose(engine.transpose(out_exp) * exp_sum)


def hinge_loss(logits, targets):
    n = logits.data.shape[0]
    return Variable(1.0/n) * engine.sum(engine.relu(Variable(1) - targets * logits))
                  

def l2(model):
    s = Variable(0)
    for p in model.parameters():
        s = s + engine.sum(p*p)
    return s
