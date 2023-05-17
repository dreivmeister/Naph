import numpy as np
class SparseMatrix:
    def __init__(self, matrix) -> None:
        self.construct_sparse_matrix(matrix)
        
        print(self.val)
        print(self.col_id)
        print(self.row_ptr)
    
    def construct_sparse_matrix(self, matrix):
        self.row_ptr = []
        self.col_id = []
        self.val = []
        
        
        k = 0
        for i in range(matrix.shape[0]):
            self.row_ptr.append(k)
            for j in range(matrix.shape[1]):
                if matrix[i,j] != 0.:
                    k += 1
                    self.val.append(matrix[i,j])
                    self.col_id.append(j)
        self.row_ptr.append(len(self.col_id))
        
    def access_element(self, i, j):
        ks = []
        for q in range(len(self.col_id)):
            if self.col_id[q] == j:
                ks.append(q)
        
        for k in ks:
            if self.row_ptr[i] <= k and self.row_ptr[i+1] > k:
                return self.val[k]
        return None
                
    
    
M = SparseMatrix(np.array([
    [10.,0.,0.,12.,0.],
    [0.,0.,11.,0.,13.],
    [0.,16.,0.,0.,0.],
    [0.,0.,11.,0.,13.],
]))


print(M.access_element(3,4))