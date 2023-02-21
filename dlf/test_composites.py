import numpy as np
import torch

import engine
import nn



def test_softmax():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.softmax(xt,dim=1)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = nn.softmax(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_softmax SUCCESS')
    else:
        print('test_softmax FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_softmax()


def test_crossentropy():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(2,3)
    t = np.array([[1,0,0],
                  [0,1,0]])
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    tt = torch.Tensor(t)
    tt.requires_grad = True
    zt = torch.nn.functional.cross_entropy(xt,tt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    tv = engine.Variable(t)
    zv = nn.cross_entropy(nn.softmax(xv),tv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_crossentropy SUCCESS')
    else:
        print('test_crossentropy FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_crossentropy()