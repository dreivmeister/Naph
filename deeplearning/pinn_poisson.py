
# imports
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation




class SurrogateNetwork(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, out_dim))
    
    def forward(self, x):
        return self.net(x)
        
    
class ResidualNetwork(nn.Module):
    def __init__(self, u, v, null):
        super().__init__()
        self.u = u
        self.v = v
        self.null = null
        self.l1 = 1.0
        self.l2 = 0.01
        self.mse = nn.MSELoss()

    def forward(self, psi, p):
        
        # encode the navier stokes
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u_t + self.l1 * (u * u_x + v * u_y) + p_x - self.l2 * (u_xx + u_yy)
        g = v_t + self.l1 * (u * v_x + v * v_y) + p_y - self.l2 * (v_xx + v_yy)
        
        # computes the residual and returns it
        u_loss = self.mse(u, self.u)
        v_loss = self.mse(v, self.v)
        f_loss = self.mse(f, self.null)
        g_loss = self.mse(g, self.null)
        return u_loss + v_loss + f_loss + g_loss
        
    


class NavierStokes():
    def __init__(self, X, Y, T, u, v):

        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)

        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

        #null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1))

        # initialize network:
        self.surr_net = SurrogateNetwork(inp_dim=3, out_dim=2)
        self.res_net = ResidualNetwork(self.u, self.v, self.null)

        self.optimizer = torch.optim.LBFGS(self.surr_net.parameters())


    def train(self):
        self.surr_net.train()
        
        for epoch in range(10):
            self.optimizer.zero_grad()
            
            # get preds
            predictions = self.surr_net(torch.hstack((self.x, self.y, self.t)))
            psi, p = predictions[:, 0:1], predictions[:, 1:2]

            # calculate losses
            loss = self.res_net(psi, p)
            
            # derivative with respect to net's weights:
            loss.backward()
            self.optimizer.step()

            if epoch % 5 == 0:
                print('Iteration: {:}, Loss: {:0.6f}'.format(epoch, loss))




if __name__=="__main__":
    # data preparation
    N_train = 5000
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    x_test = X_star[:, 0:1]
    y_test = X_star[:, 1:2]
    p_test = P_star[:, 0:1]
    u_test = U_star[:, 0:1, 0]
    t_test = np.ones((x_test.shape[0], x_test.shape[1]))

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    # input training
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    # output training
    u_train = u[idx, :]
    v_train = v[idx, :]

    
    
    # training
    pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
    pinn.train()
    torch.save(pinn.net.state_dict(), 'model.pt')
    
    

    # testing
    pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
    pinn.net.load_state_dict(torch.load('model.pt'))
    pinn.net.eval()

    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
    t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)

    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, t_test)


    # plotting
    u_plot = p_out.data.cpu().numpy()
    u_plot = np.reshape(u_plot, (50, 100))

    fig, ax = plt.subplots()

    plt.contourf(u_plot, levels=30, cmap='jet')
    plt.colorbar()

    def animate(i):
        ax.clear()
        u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i*t_test)
        u_plot = p_out.data.cpu().numpy()
        u_plot = np.reshape(u_plot, (50, 100))
        cax = ax.contourf(u_plot, levels=20, cmap='jet')
        plt.xlabel(r'$x$')
        plt.xlabel(r'$y$')
        plt.title(r'$p(x,\; y, \; t)$')

    # Call animate method
    ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
    #ani.save('p_field_lbfgs.gif')
    #plt.close()
    # Display the plot
    plt.show()