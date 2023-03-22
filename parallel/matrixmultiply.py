import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt





def square_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res


@njit(parallel=True)
def p_square_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res



if __name__=="__main__":
    serial = []
    parallel = []
    
    # run it once for compilation
    p_square_matrix_multiply(np.array([[1,2],[3,4]]),np.array([[1,2],[3,4]]))
    
    n = 100
    for i in range(1, n):
        print(i)
        A = np.random.randint(1, 50, size = (i, i))
        B = np.random.randint(1, 50, size = (i, i))

        
        s1 = time.time()
        res = square_matrix_multiply(A,B)
        e1 = time.time()
        serial.append(e1-s1)
        
        
        s2 = time.time()
        res1 = p_square_matrix_multiply(A,B)
        e2 = time.time()
        parallel.append(e2-s2)
    
    plt.plot(serial, label='serial')
    plt.plot(parallel, label='parallel')
    plt.legend()
    plt.show()