import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



def initialize_u(max_iter_time,
                 dim,
                 boundary_cond = None, 
                 initial_cond = None):
    
    if len(dim) == 1:
        dimension = (max_iter_time, dim[0])
    elif len(dim) == 2:
        dimension = (max_iter_time, dim[0], dim[1])
    
    if initial_cond[0] == 'fixed':
        u = np.full(dimension, initial_cond[1])
    elif initial_cond[0] == 'normal':
        u = np.random.normal(loc=initial_cond[1], scale=initial_cond[2], size=dimension)
        
    if boundary_cond[0] == 'dirichlet':
        if len(dim) == 1:
            u[:, 0] = boundary_cond[1] #left
            u[:, -1] = boundary_cond[2] #right
        if len(dim) == 2:
            u[:, 0, :] = boundary_cond[1] #top
            u[:, :, 0] = boundary_cond[2] #left
            u[:, -1, :] = boundary_cond[3] #bottom
            u[:, :, -1] = boundary_cond[4] #right
    elif boundary_cond[0] == 'neumann':
        raise NotImplementedError('neumann boundary not availaible')
    
    return u


def calculate_u_1d(u, f):
    nk, _ = u.shape
    for k in range(nk-1):
        result = f(u,k)
        u[k+1,1:-1] = result
    return u

# calc 2d pde
def calculate_u_2d(u, f):
    nk, _, _ = u.shape
    for k in range(nk-1):
        result = f(u,k)
        u[k+1, 1:-1, 1:-1] = result
    return u


def plot_u_1d(u):
    fig = plt.figure()
    def animate(k):
        plt.clf()
        plt.plot(u[k])
        return plt
    anim = animation.FuncAnimation(fig, animate, frames=max_iter_time, repeat=False)
    anim.save("pde_1d_solve.gif")


def plot_u_2d(u):
    fig = plt.figure()
    def animate(k):
        # Clear the current plot figure
        plt.clf()
        plt.xlabel("x")
        plt.ylabel("y")
        # This is to plot u_k (u at time-step k)
        plt.pcolormesh(u[k], cmap=plt.cm.jet, vmin=0, vmax=200)
        plt.colorbar()
        return plt
    anim = animation.FuncAnimation(fig, animate, frames=max_iter_time, repeat=False)
    anim.save("pde_2d_solve.gif")


def plot_u_3d(u):
    X = np.arange(0, plate_length, 1)
    Y = np.arange(0, plate_width, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    def animate(n):
        ax.cla()
        ax.set_zlim3d(0, 100)
        ax.plot_surface(X,Y,u[n,:,:])
        return fig,
    
    anim = animation.FuncAnimation(fig, animate, frames=max_iter_time, repeat=False)
    anim.save("pde_3d_solve.gif")    
    
class Solve:
    def __init__(self, f, length, width, steps, i_cond, b_cond, plot, *args, **kwargs) -> None:
        if width == 0:
            u_init = initialize_u(max_iter_time=steps, dim=[length], boundary_cond=b_cond, initial_cond=i_cond)
            u_result = calculate_u_1d(u_init, f)
        else:
            u_init = initialize_u(max_iter_time=steps, dim=[length, width], boundary_cond=b_cond, initial_cond=i_cond)
            u_result = calculate_u_2d(u_init, f)
        
        if plot == '1D':
            plot_u_1d(u_result)
        elif plot == '2D':
            plot_u_2d(u_result)
        elif plot == '3D':
            plot_u_3d(u_result)
        
        
        

if __name__=="__main__":
    plate_length = 50
    plate_width = 50
    max_iter_time = 50
    
    alpha = 2
    delta_x = 1
    delta_t = (delta_x ** 2)/(4 * alpha)
    gamma = (alpha * delta_t) / (delta_x ** 2)
    
    # Boundary conditions
    boundary_conditions = ['dirichlet',100.0,50.0,100.0,50.0] # top,left,bottom,right
    # Initial condition everywhere inside the grid
    initial_condition = ['fixed',50.0]
    plot = '2D'
    
    # u_init = initialize_u(max_iter_time=max_iter_time, dim=[plate_length,plate_width], boundary_cond=boundary_conditions, initial_cond=initial_condition)
    
    # # define the pde in finite diff approximation
    # def f(u,k):
    #     A = u[k, 2:  , 1:-1] #i-1,j
    #     B = u[k,  :-2, 1:-1] #i+1,j
    #     C = u[k, 1:-1, 2:  ] #i,j-1
    #     D = u[k, 1:-1,  :-2] #i,j+1
    #     E = u[k, 1:-1, 1:-1] #i,j
    #     F = u[k-1,1:-1,1:-1] #k-1,i,j
    #     return gamma * (A+B+C+D-4*E) + 2*E - F #2d wave equation
    
    # u_result = calculate_u_2d(u_init, f)
    
    
    # if plot == '1D':
    #     plot_u_1d(u_result)
    # if plot == '2D':
    #     plot_u_2d(u_result)
    # elif plot == '3D':
    #     plot_u_3d(u_result)
    # print('SAVED')
    
    
    def f(u,k):
        A = u[k, 2:  , 1:-1] #i-1,j
        B = u[k,  :-2, 1:-1] #i+1,j
        C = u[k, 1:-1, 2:  ] #i,j-1
        D = u[k, 1:-1,  :-2] #i,j+1
        E = u[k, 1:-1, 1:-1] #i,j
        F = u[k-1,1:-1,1:-1] #k-1,i,j
        return gamma * (A+B+C+D-4*E) + 2*E - F #2d wave equation
    
    s = Solve(f, plate_length, plate_width, max_iter_time, initial_condition, boundary_conditions, '2D', gamma=gamma)