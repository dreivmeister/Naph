#gauss
#lu with/without pivoting
#qr with givens or householder
#cholesky

import numpy as np

def LU_partial_decomposition(matrix):
    # lu with partial pivoting
    n, _ = matrix.shape
    #P = np.identity(n)
    #L = np.identity(n)
    U = matrix.copy()
    PF = np.identity(n)
    LF = np.zeros((n,n))
    for k in range(0, n-1):
        index = np.argmax(abs(U[k:,k])) # find abs max of curr col
        index = index + k
        
        if index != k:
            P = np.identity(n)
            P[[index,k],k:n] = P[[k,index],k:n]
            U[[index,k],k:n] = U[[k,index],k:n] 
            PF = np.dot(P,PF)
            LF = np.dot(P,LF)
        
        L = np.identity(n)
        for j in range(k+1,n):
            t = U[j,k] / U[k,k]
            L[j,k] = -t
            LF[j,k] = t
        U = np.dot(L,U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U

def back_sub(A,b):
    n = len(b)
    x = np.zeros(n)
    x[-1] = b[-1]/A[-1,-1]
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s += A[i,j]*x[j]
        x[i] = (b[i]-s)/A[i,i]
    return x

def forward_sub(A,b):
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0]/A[0,0]
    for i in range(1, n):
        s = 0
        for j in range(i):
            s += A[i,j]*x[j]
        x[i] = (b[i]-s)/A[i,i]
    return x

def LU_partial_solve(P,L,U,b):
    pb = np.dot(P,b)
    y = forward_sub(L,pb)
    x = back_sub(U,y)
    return x
    


# # Usage
# A = [[2, 1, 1, 0], [4, 6, 3, 1], [8, 7, 3, 5], [6, 7, 9, 1]]
# A = np.array(A)

# #A = np.array([[1,2],[3,4]])
# P1, L1, U1 = LU_partial_decomposition(A)
# print(P1)
# print(L1)
# print(U1)

# b = np.array([1,3,3,1])
# x = LU_partial_solve(P1,L1,U1,b)
# print(x)

# x1 = np.linalg.solve(A,b)
# print(x1)



def cholesky_decomposition(A):
    # assume for now A is symmetric and positive definite
    n = A.shape[0]
    L = np.zeros((n,n))
    
    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s += L[i,k]*L[j,k]
            
            if i == j:
                L[i,j] = np.sqrt(A[i,i]-s)
            else:
                L[i,j] = 1.0/L[j,j]*(A[i,j]-s)
    return L


# A = np.array([
#     [1,1,2,3],
#     [1,5,4,7],
#     [2,4,14,11],
#     [3,7,11,30]
# ])
# L = cholesky_decomposition(A)
# print(L)



def householder_vectorized(a):
    """Use this version of householder to reproduce the output of np.linalg.qr 
    exactly (specifically, to match the sign convention it uses)
    
    based on https://rosettacode.org/wiki/QR_decomposition#Python
    """
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)
    
    return v,tau


def qr_decomposition(A: np.ndarray):
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)
    
    for j in range(0, n):
        # Apply Householder transformation.
        v, tau = householder_vectorized(R[j:, j, np.newaxis])
        H = np.identity(m)
        H[j:, j:] -= tau * (v @ v.T)
        R = H @ R
        Q = H @ Q
    return Q[:n].T, np.triu(R[:n])

def overdetermined_linear_system_solve(A, b):
    Q,R = qr_decomposition(A)
    bs = Q@b
    x = np.linalg.solve(R,bs)
    return x

if __name__=="__main__":
    # m = 5
    # n = 3

    # A = np.random.rand(m, n)
    # print(A)
    #q, r = np.linalg.qr(A)
    #Q, R = qr_decomposition(A, snake=True)
    
    
    
    # A = np.array([[1,-1/4,1/16],[1,1/2,1/4],[1,2,4],[1,5/2,25/4]])
    # b = np.array([0,1,0,1])
    
    # print(overdetermined_linear_system_solve(A,b))
    
    # A = np.array([[3,2,1,0], 
    #               [-6,-5,2,-3], 
    #               [15,14,-16,14], 
    #               [9,8,-10,9]])
    
    # b = np.array([5,-5,-3,-2])
    
    
    # p,l,u = LU_partial_decomposition(A)
    
    # print(LU_partial_solve(p,l,u,b))
    
    A = np.array([[3,7], 
                  [0,12], 
                  [4,1]])
    
    q,r = qr_decomposition(A)
    
    print(q)
    print(r)