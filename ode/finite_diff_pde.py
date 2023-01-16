import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plate_length = 50
plate_width = 50
max_iter_time = 10

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

# Boundary conditions
boundary_conditions = [0.0,0.0,100.0,0.0] # top,left,bottom,right

# Initial condition everywhere inside the grid
initial_condition = 0.0


def initialize_u(max_iter_time, boundary_cond, initial_cond, ni=plate_length, nj=plate_width):
    
    # Initialize solution: the grid of u(k, i, j)
    u = np.full((max_iter_time, ni, nj), initial_cond)

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
        result = gamma * (A+B+C+D-4*E) + E 
        # set the newly computed heatmap at time k+1
        u[k+1, 1:-1, 1:-1] = result
    return u


# create new map and display result
u3 = calculate_u(initialize_u(max_iter_time, boundary_conditions, initial_condition, plate_length, plate_width))



def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt



# Do the calculation here
def animate(k):
    plotheatmap(u3[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("heat_equation_solution.gif")
print("Done!")