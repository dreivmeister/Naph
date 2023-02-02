import numpy as np

def GMRES(A, b, x0, nmax_iter, tol=1e-10):
    n = len(b)
    bb = np.copy(b)
    r = b - np.dot(A, x0)

    x = np.zeros((nmax_iter,n))
    q = np.zeros((nmax_iter,n))

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
        y = np.dot(A, q[k])

        # gram schmidt orthogonalize
        for j in range(k+1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
            
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros((nmax_iter + 1,))
        b[0] = np.linalg.norm(r)
        result = np.linalg.lstsq(h, b, rcond=None)[0]
        x[k] = np.dot(q.transpose(), result) + x0
        
        # potentially break early
        rr = bb - np.dot(A,x[k])
        if np.linalg.norm(rr) < tol:
            return x[k]
        
    return x[-1]


def jacobi_method(A,b,x0,tol=1e-10,max_iter=100):
    n = len(b)
    # max iterations
    for k in range(max_iter):
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j == i:
                    continue
                sigma += A[i,j]*x0[j]
            x0[i] = (b[i] - sigma) / A[i,i]
        #check early exit
        r = np.linalg.norm(b - np.dot(A,x0))
        if r < tol:
            print(f'early exit after {k} iterations!')
            return x0
    return x0


def gauss_seidel(A,b,x0,tol=1e-10,max_iter=100):
    n = len(b)
    x1 = np.copy(x0)
    # max iterations
    fehler = 0
    for i in range(max_iter):
        for k in range(n):
            # new vals
            new_sum = 0
            for j in range(k):
                new_sum += A[k,j]*x1[j]
            # old vals
            old_sum = 0
            for j in range(k+1,n):
                old_sum += A[k,j]*x0[j]
            
            x1[k] = (b[k]-new_sum-old_sum)/A[k,k]
        fehler = max(fehler,np.linalg.norm(x0-x1))
        if fehler < tol:
            print(f'early exit after {i} iterations!')
            return x1
        x0 = np.copy(x1)
    return x1
        


if __name__=='__main__':
    # initialize the matrix
    A = np.array([[10., -1., 2., 0. ],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8. ]])
    # initialize the RHS vector
    b = np.array([6., 25., -11., 15.])
    x0 = np.zeros_like(b)
    
    print(gauss_seidel(A,b,x0)) # expect 
    
    #print(GMRES(A,b,np.array([0.,0.]),10))