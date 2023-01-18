import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plate_length = 50
plate_width = 50
max_iter_time = 100

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Boundary conditions
boundary_conditions = [100.0,50.0,100.0,50.0] # top,left,bottom,right

# Initial condition everywhere inside the grid
initial_condition = 50.0


def initialize_u(max_iter_time, 
                 type_boundary,  
                 type_initial,
                 boundary_cond = None,
                 initial_cond = None, 
                 ni=plate_length, 
                 nj=plate_width):
    
    if type_initial == 'fixed':
        # Initialize solution: the grid of u(k, i, j)
        u = np.full((max_iter_time, ni, nj), initial_cond)
    elif type_initial == 'normal': # normal distribution
        u = np.random.normal(loc=0.0, scale=1.0, size=(max_iter_time, ni, nj))

    if type_boundary == 'dirichlet':
        assert boundary_cond is not None
        # Set the boundary conditions
        u[:, 0, :] = boundary_cond[0] #top
        u[:, :, 0] = boundary_cond[1] #left
        u[:, -1, :] = boundary_cond[2] #bottom
        u[:, :, -1] = boundary_cond[3] #right
        
    return u


# calc 2d pde
def calculate_u(u):
    nk, ni, nj = u.shape
    for k in range(0, nk-1):
        A = u[k, 2:  , 1:-1] #i-1,j
        B = u[k,  :-2, 1:-1] #i+1,j
        C = u[k, 1:-1, 2:  ] #i,j-1
        D = u[k, 1:-1,  :-2] #i,j+1
        E = u[k, 1:-1, 1:-1] #i,j
        F = u[k-1,1:-1,1:-1] #k-1,i,j
        #result = gamma * (A+B+C+D-4*E) + 2*E - F #2d wave equation
        result2 = E*(1*(D-E)+1) #inviscid burger equation
        # set the newly computed heatmap at time k+1
        u[k+1, 1:-1, 1:-1] = result2
    return u



def plot_u_2d(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Displacement at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

if __name__=="__main__":
    plot = '2D'
    u_init = initialize_u(max_iter_time=max_iter_time, type_boundary='dirichlet', type_initial='normal', boundary_cond=boundary_conditions, ni=plate_length, nj=plate_width)
    u_result = calculate_u(u_init)
    
    
    if plot == '2D':
        # Do the calculation here
        def animate(k):
            plot_u_2d(u_result[k], k)

        
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, animate, interval=1, frames=max_iter_time, repeat=False)
        anim.save("pde_2d_solve.gif")
        print("Done")
    elif plot == '3D':
        X = np.arange(0, plate_length, 1)
        Y = np.arange(0, plate_width, 1)
        X, Y = np.meshgrid(X, Y)
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_zlim3d(0,200)
        def animate(n):
            ax.cla()
            ax.plot_surface(X,Y,u_result[n,:,:])
            return fig,
        
        anim = animation.FuncAnimation(fig=fig,func=animate,frames=max_iter_time,repeat=False)
        anim.save("pde_3d_solve.gif")
        print("Done")
            