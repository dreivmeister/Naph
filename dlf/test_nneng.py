import torch
import numpy as np
import engine
from engine import Variable
import nn

# need to test binary_crossentropy, cross_entropy and softmax


x = torch.Tensor([[-4.0,2.0]])
x.requires_grad = True
l = torch.nn.Linear(2,4)
y = torch.nn.functional.relu(l(x))
t = torch.Tensor([[3.0,6.0,9.0,4.5]])
loss = torch.nn.functional.mse_loss(y,t)

loss.backward()
xpt, ypt = x, loss


x1 = Variable(np.expand_dims(np.array([-4.0,2.0]), axis=0))
l1 = nn.LinearLayer(2,4,Variable(l.weight.detach().numpy().T),Variable(l.bias.detach().numpy()))
y1 = engine.relu(l1.forward(x1))
t1 = Variable(np.expand_dims(np.array([3.0,6.0,9.0,4.5]), axis=0))
loss1 = nn.mean_squared_error(y1,t1)
engine.backward_graph(loss1)
xmg, ymg = x1, loss1


print(loss.data, loss1.data)
# forward pass went well
print(ymg.data, ypt.data.numpy())
# backward pass went well
print(xmg.grad, xpt.grad.numpy())