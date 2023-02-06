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
    out.data = np.clip(out.data, eps, 1. - eps)
    N = out.data.shape[0]
    ce = engine.sum(targets * engine.log(out))
    ce = Variable(-(1/N)) * ce
    return ce
    
    
def softmax(out):
    out_exp = engine.exp(out - engine.max(out))
    exp_sum = engine.sum(out_exp, ax=1)
    exp_sum.data = 1./exp_sum.data
    return engine.transpose(engine.transpose(out_exp) * exp_sum)

def hinge_loss(logits, targets):
    n = logits.data.shape[0]
    return Variable(1.0/n) * engine.sum(engine.relu(Variable(1) - targets * logits))
                
                
#TODO:
#-regularization techniques (l1,l2,dropout,batchnorm)
#-different optimiziers
#-plotting            
            

if __name__=='__main__':
    
    #quick random example
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = LinearLayer(2,32)
            self.fc2 = LinearLayer(32,100)
            self.fc3 = LinearLayer(100,3)

        def forward(self, inp):
            n1 = engine.relu(self.fc1.forward(inp))
            n2 = engine.relu(self.fc2.forward(n1))
            n3 = engine.relu(self.fc3.forward(n2))
            return softmax(n3)
        
        def parameters(self):
            return [*self.fc1.parameters()]
    
    
    m = Net()
    l = [i.data*i.data for i in m.parameters()]
    print(sum(sum(l[0])))
    
    
    
    
    # # # ALWAYS NEED THE BATCH DIMENSION!!
    # # a = Variable(np.expand_dims(np.array([1.,2.,3.]), axis=0))
    # # print(a.data.shape)
    # # l1 = LinearLayer(3,10)
    # # y = l1.forward(a)
    # # l1.zero_grad()
    # # engine.backward_graph(y)
    
    
    # # data generation
    # def f(x,y):
    #     return 3*x + 2*y
    # def g(x,y):
    #     return x
    # def h(x,y):
    #     return 6*x - y
    
    # eps = 1e-10
    # inp = []
    # target = []
    # for x in range(10):
    #     for y in range(10):
    #         inp.append([x,y])
    #         if x > 4:
    #             target.append([1.,0.,0.])
    #         elif x < 3 and y > 6:
    #             target.append([0.,1.,0.])
    #         else:
    #             target.append([0.,0.,1.])
    
    # a = Variable(np.array(inp))
    # target = Variable(np.array(target))
    # print(a.data.shape)
    # print(target.data.shape)
    
    
    
    
    
    # model = Net()
    # # training
    # for i in range(100):
    #     out = model.forward(a)
    #     loss = cross_entropy(out,target)
    #     model.zero_grad()
    #     engine.backward_graph(loss)
        
    #     if i % 10 == 0:
    #         #print(out.data)
    #         print(f"loss: {loss.data[0]}")
            
    #     model.step(3e-4)
        
        
    # # testing
    # print(model.forward(Variable(np.array((2,7)))).data)