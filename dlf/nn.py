import numpy as np
import engine
from engine import Variable

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
        return engine.plus_bcast(engine.dot(x, self.w), self.b)
    
    def parameters(self):
        return [self.w, self.b]
        
                
if __name__=='__main__':
    # quick random example
    n_hidden = [1000,100]

    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = LinearLayer(1,n_hidden[0])
            self.fc2 = LinearLayer(n_hidden[0],n_hidden[1])
            self.fc3 = LinearLayer(n_hidden[1],1)

        def forward(self, inp):
            n1 = engine.relu(self.fc1.forward(inp))
            n2 = engine.relu(self.fc2.forward(n1))
            return self.fc3.forward(n2)
        
        def parameters(self):
            return [*self.fc1.parameters(), *self.fc2.parameters(), *self.fc3.parameters()]

    model = Net()
    
    
    for i in range(100):
        a = Variable(np.random.uniform(-10,10,(100,1)))
        b = Variable(np.random.uniform(-10,10,(100,1)))
        out = model.forward(a)
        residual = engine.minus(out,b)
        loss = engine.c_mul(engine.sumel(engine.multiply(residual,residual)),1.0/100)
        model.zero_grad()
        engine.backward_graph(loss)
        
        if i % 2 == 0: 
            print(loss.data[0])
            
        model.step(1e-2)