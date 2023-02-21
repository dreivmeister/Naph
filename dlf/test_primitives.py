import numpy as np
import torch
import engine

def test_add():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    y = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    yt = torch.Tensor(y)
    yt.requires_grad = True
    zt = xt + yt
    zt.sum().backward()
    
    xv = engine.Variable(x)
    yv = engine.Variable(y)
    zv = xv + yv
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad) and np.allclose(yt.grad.numpy(),yv.grad):
        print('test_add SUCCESS')
    else:
        print('test_add FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        print(f"grad:\n{yt.grad.numpy()}\n{yv.grad}")
    
test_add()


def test_mul():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    y = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    yt = torch.Tensor(y)
    yt.requires_grad = True
    zt = xt * yt
    zt.sum().backward()
    
    xv = engine.Variable(x)
    yv = engine.Variable(y)
    zv = xv * yv
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad) and np.allclose(yt.grad.numpy(),yv.grad):
        print('test_mul SUCCESS')
    else:
        print('test_mul FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        print(f"grad:\n{yt.grad.numpy()}\n{yv.grad}")
        
test_mul()


def test_sub():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    y = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    yt = torch.Tensor(y)
    yt.requires_grad = True
    zt = xt - yt
    zt.sum().backward()
    
    xv = engine.Variable(x)
    yv = engine.Variable(y)
    zv = xv - yv
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad) and np.allclose(yt.grad.numpy(),yv.grad):
        print('test_sub SUCCESS')
    else:
        print('test_sub FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        print(f"grad:\n{yt.grad.numpy()}\n{yv.grad}")
        
test_sub()


def test_pow():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    p = np.random.randint(1,10)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = xt ** p
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = xv ** p
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_pow SUCCESS')
    else:
        print('test_pow FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_pow()


def test_div():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    y = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    yt = torch.Tensor(y)
    yt.requires_grad = True
    zt = xt / yt
    zt.sum().backward()
    
    xv = engine.Variable(x)
    yv = engine.Variable(y)
    zv = xv / yv
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad) and np.allclose(yt.grad.numpy(),yv.grad):
        print('test_div SUCCESS')
    else:
        print('test_div FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        print(f"grad:\n{yt.grad.numpy()}\n{yv.grad}")
        
test_div()


def test_neg():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = -xt
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = -xv
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_neg SUCCESS')
    else:
        print('test_neg FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_neg()


def test_sum(axis):
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    xt = torch.Tensor(x)
    xt.requires_grad = True
    if axis is None:
        zt = torch.sum(xt)
    else:
        zt = torch.sum(xt,dim=axis)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.sum(xv,ax=axis)
    print('zv',zv.data.shape)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_sum SUCCESS')
    else:
        print('test_sum FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")

# there is some weird dimension thing going on
# will have to figure that out
test_sum(0)
test_sum(1)
test_sum(None)

def test_transpose():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.transpose(xt,0,1)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.transpose(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_transpose SUCCESS')
    else:
        print('test_transpose FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_transpose()


def test_dot():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    y = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    yt = torch.Tensor(y.T)
    yt.requires_grad = True
    zt = torch.matmul(xt,yt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    yv = engine.Variable(y.T)
    zv = engine.dot(xv,yv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad) and np.allclose(yt.grad.numpy(),yv.grad):
        print('test_dot SUCCESS')
    else:
        print('test_dot FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        print(f"grad:\n{yt.grad.numpy()}\n{yv.grad}")
        
test_dot()


def test_relu():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.nn.functional.relu(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.relu(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_relu SUCCESS')
    else:
        print('test_relu FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_relu()


def test_max():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.max(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.max(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_max SUCCESS')
    else:
        print('test_max FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_max()


def test_mean():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.mean(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.mean(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data[0]) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_mean SUCCESS')
    else:
        print('test_mean FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data[0]}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_mean()


def test_variance():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.var(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.variance(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_variance SUCCESS')
    else:
        print('test_variance FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
#test_variance()


def test_exp():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.exp(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.exp(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_exp SUCCESS')
    else:
        print('test_exp FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_exp()


def test_log():
    a,b = np.random.randint(1,10,size=2)
    x = np.random.rand(a,b)
    
    xt = torch.Tensor(x)
    xt.requires_grad = True
    zt = torch.log(xt)
    zt.sum().backward()
    
    xv = engine.Variable(x)
    zv = engine.log(xv)
    engine.backward_graph(zv)
    
    
    if np.allclose(zt.data.numpy(),zv.data) and np.allclose(xt.grad.numpy(),xv.grad):
        print('test_log SUCCESS')
    else:
        print('test_log FAILED')
        print(f"out:\n{zt.data.numpy()}\n{zv.data}")
        print(f"grad:\n{xt.grad.numpy()}\n{xv.grad}")
        
test_log()




