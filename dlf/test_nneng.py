import torch
import numpy as np
import engine
from engine import Variable
import nn

# need to test binary_crossentropy, cross_entropy and softmax

def linear_layer_mse_test():
    x = torch.Tensor([[-4.0,2.0],[3.0,1.0]])
    x.requires_grad = True
    
    l = torch.nn.Linear(2,3)
    l2 = torch.nn.Linear(3,3)
    y = l2(torch.nn.functional.relu(l(x)))
    t = torch.Tensor([[3.0,9.0,4.5],[1.0,9.0,3.0]])
    loss = torch.nn.functional.mse_loss(y,t)

    loss.backward()
    xpt, ypt = x, loss


    x1 = Variable(np.array([[-4.0,2.0],[3.0,1.0]]))
    l1 = nn.LinearLayer(2,3,Variable(l.weight.detach().numpy().T),Variable(l.bias.detach().numpy()))
    l12 = nn.LinearLayer(3,3,Variable(l2.weight.detach().numpy().T),Variable(l2.bias.detach().numpy()))
    y1 = l12.forward(engine.relu(l1.forward(x1)))
    t1 = Variable(np.array([[3.0,9.0,4.5],[1.0,9.0,3.0]]))
    loss1 = nn.mean_squared_error(y1,t1)
    
    engine.backward_graph(loss1)
    xmg, ymg = x1, loss1


    if np.allclose(ymg.data,ypt.data.numpy()) and np.allclose(xmg.grad,xpt.grad.numpy()):
        print(f"linear_layer_mse_test SUCCESS!")
        return True # test success
    
    print(f"linear_layer_mse_test FAILED!")
    #print(loss.data, loss1.data)
    # forward pass went well
    print(f"loss:\ndlf: {ymg.data[0]}\ntorch: {ypt.data.numpy()}")
    # backward pass went well
    print(f"grad:\n{xmg.grad}\n{xpt.grad.numpy()}")
    
    return False
    
linear_layer_mse_test()


def linear_layer_multi_cross_entropy_test():
    x = torch.Tensor([[-4.0,2.0],
                      [3.0,1.0]]).float()
    x.requires_grad = True
    
    l = torch.nn.Linear(2,3)
    l2 = torch.nn.Linear(3,3)
    y = l2(torch.nn.functional.relu(l(x)))
    t = torch.Tensor([[1,0,0],
                      [0,1,0]])
    loss = torch.nn.functional.cross_entropy(y,t)
    loss.backward()
    xpt, ypt = x, loss
    


    x1 = Variable(np.array([[-4.0,2.0],
                            [3.0,1.0]]))
    l1 = nn.LinearLayer(2,3,Variable(l.weight.detach().numpy().T),Variable(l.bias.detach().numpy()))
    l12 = nn.LinearLayer(3,3,Variable(l2.weight.detach().numpy().T),Variable(l2.bias.detach().numpy()))
    y1 = nn.softmax(l12.forward(engine.relu(l1.forward(x1))))
    t1 = Variable(np.array([[1,0,0],
                            [0,1,0]]))
    loss1 = nn.cross_entropy(y1,t1)
    
    engine.backward_graph(loss1)
    xmg, ymg = x1, loss1


    if np.allclose(ymg.data,ypt.data.numpy()) and np.allclose(xmg.grad,xpt.grad.numpy()):
        print(f"linear_layer_multi_cross_entropy_test SUCCESS!")
        return True # test success
    
    print(f"linear_layer_multi_cross_entropy_test FAILED!")
    #print(loss.data, loss1.data)
    # forward pass went well
    print(f"loss:\ndlf: {ymg.data[0]}\ntorch: {ypt.data.numpy()}")
    # backward pass went well
    print(f"grad:\n{xmg.grad}\n{xpt.grad.numpy()}")
    
    return False

linear_layer_multi_cross_entropy_test()


def linear_layer_binary_cross_entropy_test():
    x = torch.Tensor([[4.0,2.0],
                      [3.0,-3.0]])
    x.requires_grad = True
    
    l = torch.nn.Linear(2,3)
    l2 = torch.nn.Linear(3,2)
    y = l2(torch.nn.functional.relu(l(x)))
    t = torch.Tensor([[1,0],
                      [0,1]])
    loss = torch.nn.functional.cross_entropy(y,t)
    loss.backward()
    xpt, ypt = x, loss


    x1 = Variable(np.array([[4.0,2.0],
                            [3.0,-3.0]]))
    l1 = nn.LinearLayer(2,3,Variable(l.weight.detach().numpy().T),Variable(l.bias.detach().numpy()))
    l12 = nn.LinearLayer(3,2,Variable(l2.weight.detach().numpy().T),Variable(l2.bias.detach().numpy()))
    y1 = nn.softmax(l12.forward(engine.relu(l1.forward(x1))))
    t1 = Variable(np.array([[1,0],
                            [0,1]]))
    loss1 = nn.cross_entropy(y1,t1)
    engine.backward_graph(loss1)
    xmg, ymg = x1, loss1


    if np.allclose(ymg.data, ypt.data.numpy()) and np.allclose(xmg.grad,xpt.grad.numpy()):
        print(f"linear_layer_binary_cross_entropy_test SUCCESS!")
        return True # test success
    
    print(f"linear_layer_binary_cross_entropy_test FAILED!")
    #print(loss.data, loss1.data)
    # forward pass went well
    print(f"loss:\ndlf: {ymg.data[0]}\ntorch: {ypt.data.numpy()}")
    # backward pass went well
    print(f"grad:\n{xmg.grad}\n{xpt.grad.numpy()}")
    
    return False


linear_layer_binary_cross_entropy_test()



def batchnorm_1d():
    batch_size = 5
    num_features = 3
    x = np.random.rand(batch_size,num_features)
    
    xt = torch.Tensor(x)
    bnt = torch.nn.BatchNorm1d(num_features)
    yt = bnt(xt)
    yt.sum().backward()
    ypt, xpt = yt, xt
    
    xnn = Variable(x)
    bn1d = nn.BatchNorm1D(num_features)
    ynn = bn1d.forward(xnn)
    engine.backward_graph(ynn)
    ymg, xmg = ynn, xnn
    
    if np.allclose(ymg.data, ypt.data.numpy()) and np.allclose(xmg.grad,xpt.grad.numpy()):
        print(f"batchnorm_1d SUCCESS!")
        return True # test success
    
    print(f"batchnorm_1d FAILED!")
    #print(loss.data, loss1.data)
    # forward pass went well
    print(f"loss:\ndlf: {ymg.data}\ntorch: {ypt.data.numpy()}")
    # backward pass went well
    #print(f"grad:\n{xmg.grad}\n{xpt.grad.numpy()}")
    
    return False
    
batchnorm_1d()


#NOTE: TORCH CELOSS EXPECTS LOGITS!! NOT PROBABILITIES (SOFTMAX OUTPUT)
#MY CELOSS EXPECTS PROBABILITIES


# predictionst = torch.Tensor(np.array([[1.0,1.1,2.5],
#                                       [3.2,0.2,0.9]]))

# predictions = nn.softmax(Variable(predictionst.detach().numpy()))

# # one hot
# targets = torch.Tensor(np.array([[0,0,1],
#                                  [0,0,1]]))

# x = torch.nn.functional.cross_entropy(predictionst, targets)
# print(x.data)

# x1 = nn.cross_entropy(predictions, Variable(targets.detach().numpy()))
# print(x1.data)

# from sklearn.metrics import log_loss
# x2 = log_loss(targets, predictions)
# print(x2)




# theta = Variable(np.random.rand(4, 1))
# x1 = nn.nll(Variable(predictions.detach().numpy()), Variable(targets.detach().numpy()), theta)
# print(x1.data)

# # Example of target with class indices
# # Example of target with class probabilities
# # loss = torch.nn.CrossEntropyLoss()
# # input = torch.randn(3, 5, requires_grad=True)
# # target = torch.randn(3, 5).softmax(dim=1)
# # print(target.data)
# # output = loss(input, target)
# # output.backward()