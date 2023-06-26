import numpy as np

# employs CSR - CompressedSparseRow technique
class SparseMatrix:
    def __init__(self, matrix=None, data=None, row_ind=None, col_ind=None, shape=None) -> None:
        if matrix is not None:
            if not isinstance(matrix, np.ndarray):
                raise AttributeError(f'matrix must be np.ndarray but is {type(matrix)}')
            self.shape = matrix.shape
            self.dtype = matrix.dtype
            self.frommatrixtocsr(matrix)
        elif data is not None and row_ind is not None and col_ind is not None and shape is not None:
            self.shape = shape
            self.dtype = np.float32
            self.fromdatatocsr(data, row_ind, col_ind, shape)
            
    
    def frommatrixtocsr(self, matrix):
        # Convert Numpy array to CSR matrix.
        self.row_ptr = []
        self.col_id = []
        self.val = []
        
        m,n = matrix.shape
        k = 0
        for i in range(m):
            self.row_ptr.append(k)
            for j in range(n):
                if matrix[i,j] != 0.:
                    k += 1
                    self.val.append(matrix[i,j])
                    self.col_id.append(j)
        self.row_ptr.append(len(self.col_id))
        
    def fromdatatocsr(self, data, row_ind, col_ind, shape):
        #construct matrix
        self.row_ptr = []
        self.col_id = []
        self.val = []
        
        m,n = shape
        k = 0
        for i in range(m):
            self.row_ptr.append(k)
            for j in range(n):
                found = -1
                for l in range(len(data)):
                    if row_ind[l] == i and col_ind[l] == j:
                        found = l
                        break
                if found != -1: # if exists
                    k += 1
                    self.val.append(data[found])
                    self.col_id.append(j)
        self.row_ptr.append(len(self.col_id))
        
        
    def toarray(self):
        """Convert CSR matrix to Numpy array.

        Returns:
            np.ndarray: Dense matrix.
        """

        array = np.zeros(self.shape).astype(self.dtype)
        m = self.shape[0]
        for i in range(m):
            num_vals = self.row_ptr[i + 1] - self.row_ptr[i]
            print(num_vals)
            for k in range(num_vals):
                val = self.val[self.row_ptr[i] + k]
                j = self.col_id[self.row_ptr[i] + k]
                array[i,j] = val

        return array
        
    def __getitem__(self, ind):
        if not isinstance(ind, tuple) or len(ind) != 2:
            raise AttributeError('can only access items in 2d matrix and no slicing')
        i,j = ind
        ks = []
        for q in range(len(self.col_id)):
            if self.col_id[q] == j:
                ks.append(q)
        
        for k in ks:
            if self.row_ptr[i] <= k and self.row_ptr[i+1] > k:
                return self.val[k]
        raise IndexError(f'element at {ind} does not exist')
    
    def matvecmul(self, x):
        # matrix vector product of shape MxN*Nx1 = Mx1
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise AttributeError('can only do matvec with np array with 1d vec')
        m,n = self.shape
        y = np.zeros(m)
        for i in range(m):
            for j in range(self.row_ptr[i],self.row_ptr[i+1]):
                y[i] += self.val[j] * x[self.col_id[j]]
        return y
                
    
    
# M = SparseMatrix(np.array([
#     [10.,0.,0.,12.,0.],
#     [0.,0.,11.,0.,13.],
#     [0.,16.,0.,0.,0.],
#     [0.,0.,11.,0.,13.],
# ]))
# print(M.toarray())

# #+x = np.array([2.,3.,0.,1.,6.])


# M1 = SparseMatrix(row_ind=[0,0,1,1,2,3,3],
#                   col_ind=[0,3,2,4,1,2,4],
#                   data=[10.,12.,11.,13.,16.,11.,13.],shape=(4,5))
# print(M1.toarray())


#print(M.matvec(M.matvec(x)))


def adjlisttoadjmat(adj_list):
    # adj_list is a list of lists of len = N
    # nodes are numbered between 0 and N-1
    N = len(adj_list)
    adj_mat = np.zeros((N,N)).astype(np.int32)
    
    for i in range(N):
        for j in adj_list[i]:
            adj_mat[i,j] = 1
    
    return adj_mat
    

#pagerank
# N - number of nodes (0, N-1)
if __name__=='__main__':
    # # default adj matrix
    # M = np.array([[0, 0, 0, 0, 1],
    #               [1, 0, 0, 0, 0],
    #               [1, 0, 0, 0, 0],
    #               [0, 1, 1, 0, 0],
    #               [0, 0, 1, 1, 0]])
    M = np.random.randint(2, size=(10,10))
    
    n = M.shape[1] # num nodes
    d = 0.85 # damping factor
    v = np.ones(n) / n
    
    # normalize each column
    row_sums = M.sum(axis=0)
    M = M / row_sums[np.newaxis, :]
    
    # apply dampening
    M = (d * M + (1 - d) / n)
    
    # computing pagerank of M using Sparse
    M_S = SparseMatrix(matrix=M)
    for i in range(20):
        v = M_S.matvecmul(v)
    print(v)