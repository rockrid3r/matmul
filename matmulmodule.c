#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include "structmember.h"

#define RAISE_EXCEPTION_IF(condition, exception, format, ...) \
    do { \
        if ((condition)) { \
            char err[256]; \
            snprintf(err, sizeof(err), format __VA_OPT__(,) __VA_ARGS__); \
            PyErr_SetString(exception, err); \
            return NULL; \
        } \
    } while (0)

static void* AllocateArray(size_t size) {
    void* data;
    if (!(data = calloc(size, sizeof(double*)))) {
        return NULL;
    }
    return data;
}

typedef struct {
    int64_t value;
} RefCnt;

typedef struct {
    PyObject_HEAD
    size_t nrows;
    size_t ncols;
    size_t rows_stride;
    size_t cols_stride;
    RefCnt* data_refcnt;
    double* data;
} MatrixObject;

#define REFCNT(matrix) \
    ((matrix)->data_refcnt)

static PyTypeObject MatrixType;

static MatrixObject* CreateUninitializedMatrix() {
    MatrixObject* matrix = (MatrixObject*) MatrixType.tp_alloc(&MatrixType, 0);
    return matrix;
}

static size_t DataIndexFromCoords(MatrixObject* matrix, size_t i, size_t j) {
    return matrix->rows_stride * i + matrix->cols_stride * j;
}

static size_t DataSize(MatrixObject* matrix) {
    return matrix->nrows * matrix->ncols;
}

static void SetRefCnt(MatrixObject* matrix, int64_t value) {
    RefCnt* refcnt = REFCNT(matrix);
    refcnt->value = value;
}

[[maybe_unused]]
static int64_t GetRefCnt(MatrixObject* matrix) {
    RefCnt* refcnt = REFCNT(matrix);
    return refcnt->value;
}

static int64_t IncrementRefCnt(MatrixObject* matrix) {
    RefCnt* refcnt = REFCNT(matrix);
    return ++(refcnt->value);
}

static int64_t DecrementRefCnt(MatrixObject* matrix) {
    RefCnt* refcnt = REFCNT(matrix);
    return --(refcnt->value);
}

static void InitMatrix(MatrixObject* matrix, size_t nrows, size_t ncols) {
    matrix->nrows = nrows;
    matrix->ncols = ncols;
    matrix->rows_stride = ncols;
    matrix->cols_stride = 1;
    matrix->data = (double*)AllocateArray(DataSize(matrix));
    matrix->data_refcnt = malloc(sizeof(RefCnt));
    SetRefCnt(matrix, 1);
}


static MatrixObject* CreateMatrixFromSize(size_t nrows, size_t ncols) {
    MatrixObject* matrix = CreateUninitializedMatrix();
    InitMatrix(matrix, nrows, ncols);
    return matrix;
}

static void FillMatrix(MatrixObject* matrix, double value) {
    for (size_t i = 0; i < DataSize(matrix); ++i) {
        matrix->data[i] = value;
    }
}

static MatrixObject* CreateViewToMatrix(MatrixObject* matrix) {
    assert(GetRefCnt(matrix) > 0);
    MatrixObject* another_view = CreateUninitializedMatrix();
    another_view->nrows = matrix->nrows;
    another_view->ncols = matrix->ncols;
    another_view->rows_stride = matrix->rows_stride;
    another_view->cols_stride = matrix->cols_stride;
    another_view->data = matrix->data;
    another_view->data_refcnt = matrix->data_refcnt;
    IncrementRefCnt(another_view);
    return another_view;
}

static MatrixObject* CreateTransposeView(MatrixObject* matrix) {
    MatrixObject* transpose_view = CreateViewToMatrix(matrix);
    transpose_view->nrows = matrix->ncols;
    transpose_view->ncols = matrix->nrows;
    transpose_view->rows_stride = matrix->cols_stride;
    transpose_view->cols_stride = matrix->rows_stride;
    return transpose_view;
}

static void Matrix_dealloc(MatrixObject* self) {
    if (DecrementRefCnt(self) == 0) {
        free(self->data);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Matrix_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    MatrixObject* self = CreateUninitializedMatrix();
    return (PyObject*)self;
}

static int Matrix_init(MatrixObject* self, PyObject* args, PyObject* kwds) {
    int nrows;
    int ncols;
    if (!PyArg_ParseTuple(args, "ii", &nrows, &ncols)) {
        Py_DECREF(self);
        return -1;
    }
    if (nrows <= 0 || ncols <= 0) {
        return -1;
    }
    InitMatrix(self, nrows, ncols);
    FillMatrix((MatrixObject*)self, 0);
    return 0;
}

static PyObject* Matrix_get(MatrixObject* self, PyObject* args) {
    int i;
    int j;
    if (!PyArg_ParseTuple(args, "ii", &i, &j)) {
        return NULL;
    }
    return PyFloat_FromDouble(self->data[DataIndexFromCoords(self, i, j)]);
}

static PyObject* Matrix_set(MatrixObject* self, PyObject* args) {
    int i;
    int j;
    double value;
    if (!PyArg_ParseTuple(args, "iid", &i, &j, &value)) {
        return NULL;
    }
    self->data[DataIndexFromCoords(self, i, j)] = value;
    Py_RETURN_NONE;
}


static void MatMulToDest(MatrixObject* A, MatrixObject* B, MatrixObject* dest) {
    for (size_t i = 0; i < A->nrows; ++i) {
        for (size_t j = 0; j < B->ncols; ++j) {
            for (size_t k = 0; k < A->ncols; ++k) {
                size_t dest_i_j = DataIndexFromCoords(dest, i, j);
                size_t A_i_k = DataIndexFromCoords(A, i, k);
                size_t B_k_j = DataIndexFromCoords(B, k, j);
                dest->data[dest_i_j] += A->data[A_i_k] * B->data[B_k_j];
            }
        }
    }
}

static MatrixObject* MatMulImplCPU(MatrixObject* A, MatrixObject* B) {
    MatrixObject* C = CreateMatrixFromSize(A->nrows, B->ncols);
    if (!C) {
        return NULL;
    }
    MatMulToDest(A, B, C);
    return C;
}

static MatrixObject* MatMulImpl(MatrixObject* A, MatrixObject* B) {
    if (A->ncols != B->nrows) {
        char err[256];
        snprintf(err, sizeof(err), 
            "Wrong shapes while matmul: (%lu, %lu) @ (%lu, %lu)", A->nrows, A->ncols, B->nrows, B->ncols);
        PyErr_SetString(PyExc_RuntimeError, err);
        return NULL;
    }
    return MatMulImplCPU(A, B);
}
 
static PyObject* Matrix_matmul(PyObject* self, PyObject* args) {
    MatrixObject* other;
    if (!PyArg_ParseTuple(args, "O!", (PyObject*) &MatrixType, (PyObject*)&other)) {
        return NULL;
    }
    return (PyObject*)MatMulImpl((MatrixObject*)self, other);
}

static MatrixObject* MatrixCopy(MatrixObject* matrix) {
    MatrixObject* copy = CreateMatrixFromSize(matrix->nrows, matrix->ncols);
    copy->rows_stride = matrix->rows_stride;
    copy->cols_stride = matrix->cols_stride;
    for (size_t i = 0; i < DataSize(matrix); ++i) {
        copy->data[i] = matrix->data[i];
    }
    return copy;
}

static PyObject* Matrix_copy(PyObject* self, PyObject* _noargs) {
    MatrixObject* self_matrix = (MatrixObject*)self;
    MatrixObject* copy = MatrixCopy(self_matrix);
    return (PyObject*)copy;
}

static PyObject* Matrix_transpose(PyObject* self, PyObject* _noargs) {
    MatrixObject* self_matrix = (MatrixObject*)self;
    MatrixObject* transpose = CreateTransposeView(self_matrix);
    return (PyObject*)transpose;
}

static PyObject* Matrix_fill(PyObject* _self, PyObject* args) {
    double value;
    RAISE_EXCEPTION_IF(!PyArg_ParseTuple(args, "d", &value),
        PyExc_ValueError, "Expected @value to be a float");
    MatrixObject* self = (MatrixObject*)_self;
    FillMatrix(self, value);
    Py_RETURN_NONE;
}

static PyMemberDef Matrix_members[] = {
    {"nrows", T_INT, offsetof(MatrixObject, nrows), 0, "Number of rows in matrix"},
    {"ncols", T_INT, offsetof(MatrixObject, ncols), 0, "Number of cols in matrix"},
    {NULL}
};

static PyMethodDef Matrix_methods[] = {
    {"get", (PyCFunction) Matrix_get, METH_VARARGS},
    {"matmul", (PyCFunction) Matrix_matmul, METH_VARARGS},
    {"set", (PyCFunction) Matrix_set, METH_VARARGS},
    {"copy", (PyCFunction) Matrix_copy, METH_NOARGS},
    {"transpose", (PyCFunction) Matrix_transpose, METH_NOARGS},
    {"fill", (PyCFunction) Matrix_fill, METH_VARARGS},
    {NULL}
};


#define VERIFY_IS_MATRIX(x, err_string) \
    RAISE_EXCEPTION_IF(!PyObject_TypeCheck(x, &MatrixType), \
        PyExc_ValueError, (err_string))

#define CHECK_BOTH_MATRICES_OF_SAME_SHAPE \
    VERIFY_IS_MATRIX(_left, "Expected @left to be a Matrix"); \
    MatrixObject* left = (MatrixObject*)_left; \
    VERIFY_IS_MATRIX(_right, "Expected @right to be a Matrix"); \
    MatrixObject* right = (MatrixObject*)_right; \
    RAISE_EXCEPTION_IF(left->nrows != right->nrows || left->ncols != right->ncols, \
        PyExc_ValueError, \
        "Shapes of @left and @right should be the same, got: (%lu, %lu) and (%lu, %lu)", \
        left->nrows, left->ncols, right->nrows, right->ncols);

static MatrixObject* MatrixAdd(MatrixObject* left, MatrixObject* right) {
    MatrixObject* result = CreateMatrixFromSize(left->nrows, left->ncols);
    for (size_t i = 0; i < left->nrows; ++i) {
        for (size_t j = 0; j < left->ncols; ++j) {
            result->data[DataIndexFromCoords(result, i, j)] = 
                left->data[DataIndexFromCoords(left, i, j)] + right->data[DataIndexFromCoords(right, i, j)];
        }
    }
    return result;
}

static PyObject* Matrix_add(PyObject* _left, PyObject* _right) {
    CHECK_BOTH_MATRICES_OF_SAME_SHAPE
    return (PyObject*)MatrixAdd(left, right);
}

static MatrixObject* MatrixSubtract(MatrixObject* left, MatrixObject* right) {
    MatrixObject* result = CreateMatrixFromSize(left->nrows, left->ncols);
    for (size_t i = 0; i < left->nrows; ++i) {
        for (size_t j = 0; j < left->ncols; ++j) {
            result->data[DataIndexFromCoords(result, i, j)] = 
                left->data[DataIndexFromCoords(left, i, j)] - right->data[DataIndexFromCoords(right, i, j)];
        }
    }
    return result;
}

static PyObject* Matrix_subtract(PyObject* _left, PyObject* _right) {
    CHECK_BOTH_MATRICES_OF_SAME_SHAPE
    return (PyObject*)MatrixSubtract(left, right);
}

static MatrixObject* MatrixMultiply(MatrixObject* left, MatrixObject* right) {
    MatrixObject* result = CreateMatrixFromSize(left->nrows, left->ncols);
    for (size_t i = 0; i < left->nrows; ++i) {
        for (size_t j = 0; j < left->ncols; ++j) {
            result->data[DataIndexFromCoords(result, i, j)] = 
                left->data[DataIndexFromCoords(left, i, j)] * right->data[DataIndexFromCoords(right, i, j)];
        }
    }
    return result;
}

static PyObject* Matrix_multiply(PyObject* _left, PyObject* _right) {
    CHECK_BOTH_MATRICES_OF_SAME_SHAPE
    PyObject* result = (PyObject*)MatrixMultiply(left, right);
    return result;
}

static MatrixObject* MatrixDivide(MatrixObject* left, MatrixObject* right) {
    MatrixObject* result = CreateMatrixFromSize(left->nrows, left->ncols);
    for (size_t i = 0; i < left->nrows; ++i) {
        for (size_t j = 0; j < left->ncols; ++j) {
            result->data[DataIndexFromCoords(result, i, j)] =
                left->data[DataIndexFromCoords(left, i, j)] / right->data[DataIndexFromCoords(right, i, j)];
        }
    }
    return result;
}

static PyObject* Matrix_divide(PyObject* _left, PyObject* _right) {
    CHECK_BOTH_MATRICES_OF_SAME_SHAPE
    return (PyObject*)MatrixDivide(left, right);
}
    
static MatrixObject* MatrixNegative(MatrixObject* matrix) {
    MatrixObject* result = CreateMatrixFromSize(matrix->nrows, matrix->ncols);
    for (size_t i = 0; i < DataSize(matrix); ++i) {
        result->data[i] = -matrix->data[i];
    }
    return result;
}

static PyObject* Matrix_negative(PyObject* _matrix) {
    MatrixObject* matrix = (MatrixObject*) _matrix;
    return (PyObject*)MatrixNegative(matrix);
}

static PyObject* Matrix_positive(PyObject* _matrix) {
    MatrixObject* matrix = (MatrixObject*) _matrix;
    return (PyObject*)MatrixCopy(matrix);
}

static PyNumberMethods Matrix_as_number_methods = {
    .nb_add = Matrix_add,
    .nb_subtract = Matrix_subtract,
    .nb_multiply = Matrix_multiply,
    .nb_true_divide = Matrix_divide,
    .nb_negative = Matrix_negative,
    .nb_positive = Matrix_positive,
};

static PyTypeObject MatrixType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matmul.Matrix",
    .tp_doc = PyDoc_STR("Matrix objects"),
    .tp_basicsize = sizeof(MatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Matrix_new,
    .tp_dealloc = (destructor)Matrix_dealloc,
    .tp_init = (initproc)Matrix_init,
    .tp_members = Matrix_members,
    .tp_methods = Matrix_methods,
    .tp_as_number = &Matrix_as_number_methods,
};

PyObject* matmul_zeros(PyObject* self, PyObject* args) {
    int nrows;
    int ncols;
    if (!PyArg_ParseTuple(args, "(ii)", &nrows, &ncols)) {
        return NULL;
    }
    MatrixObject* matrix = CreateMatrixFromSize(nrows, ncols);
    FillMatrix(matrix, 0);
    return (PyObject*)matrix;
}

PyObject* matmul_ones(PyObject* self, PyObject* args) {
    int nrows;
    int ncols;
    if (!PyArg_ParseTuple(args, "(ii)", &nrows, &ncols)) {
        return NULL;
    }
    MatrixObject* matrix = CreateMatrixFromSize(nrows, ncols);
    FillMatrix(matrix, 1);
    return (PyObject*)matrix;
}

PyObject* matmul_full(PyObject* self, PyObject* args) {
    int nrows;
    int ncols;
    double value;
    if (!PyArg_ParseTuple(args, "(ii)d", &nrows, &ncols, &value)) {
        return NULL;
    }
    MatrixObject* matrix = CreateMatrixFromSize(nrows, ncols);
    FillMatrix(matrix, value);
    return (PyObject*)matrix;
}

PyObject* matmul_seed(PyObject* self, PyObject* args) {
    int seed;
    if (!PyArg_ParseTuple(args, "(i)", &seed)) {
        return NULL;
    }
    srand((unsigned int)seed);
    Py_RETURN_NONE;
}

PyObject* matmul_rand(PyObject* self, PyObject* args) {
    int nrows;
    int ncols;
    if (!PyArg_ParseTuple(args, "(ii)", &nrows, &ncols)) {
        return NULL;
    }
    MatrixObject* matrix = CreateMatrixFromSize(nrows, ncols);
    for (size_t i = 0; i < DataSize(matrix); ++i) {
        matrix->data[i] = (double)rand() / (double)RAND_MAX;
    }
    return (PyObject*)matrix;
}

static PyMethodDef MatMulMethods[] = {
    {"zeros", matmul_zeros, METH_VARARGS, "Create matrix of given shape with each entry set to 0"},
    {"ones", matmul_ones, METH_VARARGS, "Create matrix of given shape with each entry set to 1"},
    {"full", matmul_full, METH_VARARGS, "Create matrix of given shape with each entry set to given value"},
    {"rand", matmul_rand, METH_VARARGS, "Create matrix of given shape with each entry sampled uniformly on [0, 1)"},
    {"seed", matmul_seed, METH_VARARGS, "Seed the pseudo-random number generator"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef matmulmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "matmul",
    .m_doc = "Module which implements matrix multiplication.",
    .m_methods = MatMulMethods,
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_matmul(void) {
    PyObject* m;
    if (PyType_Ready(&MatrixType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&matmulmodule);
    if (m == NULL) {
        return NULL;
    }
    if (PyModule_AddObjectRef(m, "Matrix", (PyObject*) &MatrixType) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
