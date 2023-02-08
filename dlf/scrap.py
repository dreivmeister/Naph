

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
    
    


batch_size = 5
num_features = 3
x = Variable(np.random.rand(batch_size,num_features))


# drop = Dropout()
# y = drop.forward(x)

print(x.data)

bn1d = BatchNorm1D(num_features)
y = bn1d.forward(x)

print(y.data)

engine.backward_graph(y)