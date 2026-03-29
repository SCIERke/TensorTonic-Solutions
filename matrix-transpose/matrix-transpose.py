import numpy as np
import math

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    n_row = A.shape[0]
    n_col = A.shape[1]
    
    B = np.empty((n_col, n_row))

    for i in range(n_row):
        for j in range(n_col):
            B[j][i] = A[i][j]


    return B