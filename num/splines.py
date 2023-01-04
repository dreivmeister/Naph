import numpy as np

def compute_coeffs(moments, y, h):
    coeffs = []
    for j in range(len(y)-1):
        coeffs_j = [] # aj, bj, cj, dj
        coeffs_j.append(y[j]) #aj
        coeffs_j.append((y[j+1]-y[j])/h[j]-(h[j]/6)*(moments[j+1]+2*moments[j])) #bj
        coeffs_j.append(moments[j]/2) #cj
        coeffs_j.append((moments[j+1]-moments[j])/(6*h[j]))

        coeffs.append(coeffs_j)
    return coeffs

def compute_rhs(y, h):
    rhs = []
    for j in range(1,len(y)-1):
        rhs.append((6*(y[j+1]-y[j]))/h[j]-(6*(y[j]-y[j-1]))/h[j-1])
    return rhs
        
def compute_h(x):
    h = []
    for i in range(len(x)-1):
        h.append(x[i+1]-x[i])
    return h

def fill_matrix(n, h):
    d = np.zeros((n-1,n-1))
    
    if n-1==1:
        d[0,0] = 2*(h[0]*h[1])
    
    else:
        # fill diagonal
        for i in range(d.shape[0]-1):
            d[i,i] = 2*(h[i]+h[i+1])
        
        print(d.shape[0])
        for i in range(d.shape[0]-1):
            d[1+i,i] = h[i+1]
            d[i,1+i] = h[i+1]
    
    return d
            
    
    

def cubic_spline_interpolation(x,y):
    # cond is either natural, complete or None, then it is not-a-knot
    n = len(x)-1
    print(n)
    h = compute_h(x)
    print(h)
    rhs = compute_rhs(y, h)
    print(rhs)
    d = fill_matrix(n, h)
    print(d)
    print(d.shape, len(rhs))
    moments = np.linalg.solve(d,rhs)
    print(moments)
    coeffs = compute_coeffs(moments, y, h)
    return coeffs


c = cubic_spline_interpolation([1,3,4],[2,4,2])
print(c) # [[2,2,0,-0.25],[4,-1,-1.5,0.5]]