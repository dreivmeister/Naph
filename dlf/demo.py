import random
import numpy as np
import matplotlib.pyplot as plt

import engine
from engine import Variable
import nn

from sklearn.datasets import make_moons

# load data
X, y = make_moons(n_samples=100, noise=0.1)
y = y*2 -1
# plt.figure(figsize=(5,5))
# plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap='jet')
# plt.show()

# init the model
class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = nn.LinearLayer(2,16)
            self.fc2 = nn.LinearLayer(16,16)
            self.fc3 = nn.LinearLayer(16,1)
            self.d1 = nn.Dropout()
            self.d2 = nn.Dropout()

        def forward(self, inp):
            n1 = engine.relu(self.fc1.forward(inp))
            n12 = self.d1.forward(n1)
            n2 = engine.relu(self.fc2.forward(n12))
            n21 = self.d2.forward(n2)
            n3 = self.fc3.forward(n21)
            return n3
        
        def parameters(self):
            return [*self.fc1.parameters(),*self.fc2.parameters(),*self.fc3.parameters()]



model = Classifier()
batch_size = None
# training
for i in range(100):
    # data load
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    
    # forward the model
    inputs = Variable(Xb)
    scores = model.forward(inputs)
    
    # calc the loss
    yb = Variable(np.expand_dims(yb, axis=1))
    data_loss = nn.hinge_loss(scores, yb)
    
    alpha = Variable(1e-4)
    reg_loss = nn.l2(model)
    
    total_loss = data_loss + reg_loss
    
    model.zero_grad()
    engine.backward_graph(total_loss)
    
    learning_rate = 1.0 - 0.9*i/100
    model.step(learning_rate)
    
    if i % 10 == 0:
        print(f"loss: {total_loss.data[0]}")
    


# plot the results
# visualize the decision boundary
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]

scores = []
for p in Xmesh:
    scores.append(model.forward(Variable(np.expand_dims(p,axis=0))))
scores = np.array(scores)

#print(scores[0].data)
Z = np.array([s.data.mean() > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()