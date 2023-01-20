import torch
import torch.nn as nn
from math import sin, cos, pi

#implement a loss function for PINN
def pinn_loss(t, net, g):
    # f is the pde target
    # g is the NN function
    # which needs to be diff'd 
    eps = float(1e-10)
    s = 0
    # t is time linspace
    # mean squared error
    for i in t:
        s += (((net(i+eps)-net(i))/eps) - g(i))**2
    s /= len(t)
    return s
        
class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        net = []
        for i in range(len(config)-1):
            net.append(nn.Linear(in_features=config[i],out_features=config[i+1]))
            net.append(nn.Tanh())
        self.net = nn.Sequential(*net[:-1])
    
    def forward(self, t):
        return self.net(t)


network = Net(config=[1,32,32,1])


def gen_batch(batch_size, input_dim, r1, r2):
    return (r1 - r2) * torch.rand(batch_size, input_dim) + r2

batch_size = 64
input_dim = 1
r1 = 0
r2 = 6

# rhs of ode
def g(t):
    return cos(2*pi*t)

def gg(t):
    return 1/(2*pi)*sin(2*pi*t)

data = gen_batch(batch_size,input_dim,r1,r2)
num_epochs = 1
optimizer = torch.optim.Adam(network.parameters(),lr=0.01)
for e in range(1,num_epochs+1):
    #data = gen_batch(batch_size,input_dim,r1,r2)
    optimizer.zero_grad()
    loss = pinn_loss(data, network, g)
    loss.backward()
    optimizer.step()
    if e % 100 == 0:
        print(f"epoch: {e} loss: {loss.item()}")
    
    
def plot_result(net, g):
    import matplotlib.pyplot as plt
    t = torch.linspace(r1,r2,100)
    t = torch.unsqueeze(t,1)
    net_y = net(t).detach().numpy()
    g_y = [g(i) for i in t]
    
    plt.plot(net_y, label='net')
    plt.plot(g_y, label='real')
    plt.legend(loc='upper left')
    plt.show()
    
plot_result(network, gg)
    