import numpy as np
from polynomials import horners_method

def compute_coeffs(moments, y, h):
    coeffs = []
    for j in range(len(y)-1):
        coeffs_j = [] # aj, bj, cj, dj
        coeffs_j.append(y[j]) #aj
        coeffs_j.append((y[j+1]-y[j])/h[j]-(h[j]/6)*(moments[j+1]+2*moments[j])) #bj
        coeffs_j.append(moments[j]/2) #cj
        coeffs_j.append((moments[j+1]-moments[j])/(6*h[j])) #dj

        coeffs.append(coeffs_j)
    return coeffs
        
def compute_h(x):
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    return h

def fill_matrix(h, v):
    n = len(v) # 2<-n-1 3<-n 4<-number of points
    A = np.zeros((n,n))
    
    for i in range(n):
        A[i,i] = v[i]
    for i in range(n-1):
        A[i+1,i] = h[i]
        A[i,i+1] = h[i]
    return A
    
def compute_b(y, h):
    b = []
    for i in range(len(y)-1):
        b.append((1/h[i])*(y[i+1]-y[i]))
    return b


def compute_v(h):
    v = []
    for i in range(1,len(h)):
        v.append(2*(h[i-1]+h[i]))
    return v

def compute_u(b):
    u = []
    for i in range(1,len(b)):
        u.append(6*(b[i]-b[i-1]))
    return u


def TDMA(A,b):
    # elim
    for i in range(1, A.shape[0]):
        m = A[i,i-1]/A[i-1,i-1]
        A[i,i] = A[i,i] - m*A[i-1,i]
        b[i] = b[i] - m*b[i-1]
    # back sub
    x = [b[-1]/A[-1,-1]]
    for i in range(A.shape[0]-2, -1, -1):
        x.append((b[i]-A[i,i+1]*x[-1])/A[i,i])
    return x
        
    



def cubic_spline_interpolation(x,y):
    h = compute_h(x)
    b = compute_b(y, h)
    v = compute_v(h)
    u = compute_u(b) # rhs
    A = fill_matrix(h, v)
    #print(A)
    z = TDMA(A, u)
    z.insert(0,0)
    z.append(0)
    z = z[::-1]
    coeffs = compute_coeffs(z, y, h)
    return coeffs
    

c = cubic_spline_interpolation([0.9,1.3,1.9,2.1],[1.3,1.5,1.85,2.1])
#print(c) # [[2,2,0,-0.25],[4,-1,-1.5,0.5]]


import matplotlib.pyplot as plt
def plot_cubic_spline(coeffs, nodes, domains):
    ys = [horners_method(c[::-1], domains[i], shift=nodes[i]) for i,c in enumerate(coeffs)]
    
    # plot nodes
    y_nodes = []
    for i in range(len(nodes)-1):
        y_nodes.append(horners_method(coeffs[i][::-1],nodes[i],shift=nodes[i]))
    y_nodes.append(horners_method(coeffs[-1][::-1],nodes[-1],nodes[-2]))
    
    for i,y in enumerate(ys):
        plt.plot(domains[i],y)
    plt.scatter(nodes,y_nodes)
    plt.show()
    
    
plot_cubic_spline(c, [0.9,1.3,1.9,2.1], [np.linspace(0.9,1.3,50),np.linspace(1.3,1.9,50),np.linspace(1.9,2.1,50)])