import matmul

def print_matrix(matrix):
    for i in range(matrix.nrows):
        for j in range(matrix.ncols):
            print(A.get(i, j), end = ' ')
        print()

def is_close(A, B, atol=1e-7):
    assert A.nrows == B.nrows and A.ncols == B.ncols
    for i in range(A.nrows):
        for j in range(A.ncols):
            if abs(A.get(i, j) - B.get(i, j)) > atol:
                return False
    return True

def py_matmul(A, B):
    assert A.ncols == B.nrows
    C = matmul.zeros((A.nrows, B.ncols))
    for i in range(C.nrows):
        for j in range(C.ncols):
            for k in range(A.ncols):
                C.set(i, j, C.get(i, j) + A.get(i, k) * B.get(k, j))
    return C

A = matmul.rand((5, 5))
A.set(0, 0, 1)
print_matrix(A)

assert is_close(A.matmul(A), py_matmul(A, A))
