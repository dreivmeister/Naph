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
    def __init__(self, features_in, features_out):
        super(LinearLayer, self).__init__()
        std = 1.0/features_in
        self.w = Variable(np.random.uniform(-std, std, (features_in, features_out)))
        self.b = Variable(np.random.uniform(-std, std, features_out))

    def forward(self, x):
        return engine.add(engine.dot(x, self.w), self.b)
    
    def parameters(self):
        return [self.w, self.b]



# misc functions and classes like:
# losses, optimizers, convencience utilities and so on

# regression loss
def mean_squared_error(out, target):
    # shapes have to match
    
    # get the batch dimension
    n = out.data.shape[0]
    r = engine.minus(out,target)
    return engine.const_multiply(engine.sum(engine.multiply(r,r)),1.0/n)


# binary classification loss
def binary_crossentropy(out, target):
    """
    out has the probs of being label 0
    out =    [0.2,0.4,0.9]
    target = [1,0,0]
    """
    # for numerical stability
    out.data += 1e-10
    n = out.data.shape[0]
    res = engine.plus(engine.multiply(target,engine.log(out)), 
                      engine.multiply(engine.minus(Variable(np.array([1])),target), engine.log(engine.minus(Variable(np.array([1])),out))))
    return engine.const_multiply(res, -(1./n))
    

# multi class classification
def cross_entropy(out, target):
    pass


def softmax(out):
    out_exp = engine.exp(out)
    exp_sum = engine.sum(out_exp)
    exp_sum.data = 1./exp_sum.data
    
    return engine.multiply(out_exp, exp_sum)
    

                
if __name__=='__main__':
    #quick random example
    
    
    

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = LinearLayer(2,32)
            self.fc2 = LinearLayer(32,100)
            self.fc3 = LinearLayer(100,1)

        def forward(self, inp):
            n1 = engine.relu(self.fc1.forward(inp))
            n2 = engine.relu(self.fc2.forward(n1))
            n3 = engine.relu(self.fc3.forward(n2))
            return n3
        
        def parameters(self):
            return [*self.fc1.parameters(),*self.fc2.parameters(),*self.fc3.parameters()]
        
    # a = Variable(np.random.uniform(-10,10,(1,10)))
    # print(a.data.shape)
    # y = model.forward(a)
    # print(y.data.shape)
    # model.zero_grad()
    # engine.backward_graph(y)
    
    
    
    
    # # ALWAYS NEED THE BATCH DIMENSION!!
    # a = Variable(np.expand_dims(np.array([1.,2.,3.]), axis=0))
    # print(a.data.shape)
    # l1 = LinearLayer(3,10)
    # y = l1.forward(a)
    # l1.zero_grad()
    # engine.backward_graph(y)
    
    
    # data generation
    def f(x,y):
        return 3*x + 2*y
    
    eps = 1e-10
    inp = []
    target = []
    for x in range(10):
        for y in range(10):
            inp.append((x,y))
            if (x < 5 and y < 5) or (x >= 5 and y >= 5):
                target.append(0)
            else:
                target.append(1)
    
    a = Variable(np.array(inp))
    b = Variable(np.expand_dims(np.array(target),axis=1))
    print(a.data.shape)
    print(b.data.shape)
    
    
    
    
    
    model = Net()
    # training
    for i in range(100+1):
        out = model.forward(a)
        print(softmax(out).data)
        loss = binary_crossentropy(out,b)
        model.zero_grad()
        engine.backward_graph(loss)
        
        if i % 100 == 0:
            print(loss.data[0])
            
        model.step(3e-4)
        
        
    # testing
    print(model.forward(Variable(np.array((2,7)))).data)