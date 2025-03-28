#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define IDX(i, j, cols) ((i) * (cols) + (j))

typedef struct {
    int i, j;
} Point;

void reduce_by_half(double* x, int len, double* out, int* new_len) {
    int half_len = len / 2;
    for (int i = 0; i < half_len; i++) {
        out[i] = (x[2 * i] + x[2 * i + 1]) / 2.0;
    }
    *new_len = half_len;
}

int compute_dtw_with_path(double* x, int n, double* y, int m, Point* path) {
    double* D = (double*)malloc(n * m * sizeof(double));
    int* back = (int*)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++) D[i] = INFINITY;
    D[IDX(0, 0, m)] = fabs(x[0] - y[0]);
    back[IDX(0, 0, m)] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double cost = fabs(x[i] - y[j]);
            if (i > 0 && D[IDX(i, j, m)] > D[IDX(i - 1, j, m)] + cost) {
                D[IDX(i, j, m)] = D[IDX(i - 1, j, m)] + cost;
                back[IDX(i, j, m)] = 1;
            }
            if (j > 0 && D[IDX(i, j, m)] > D[IDX(i, j - 1, m)] + cost) {
                D[IDX(i, j, m)] = D[IDX(i, j - 1, m)] + cost;
                back[IDX(i, j, m)] = 2;
            }
            if (i > 0 && j > 0 && D[IDX(i, j, m)] > D[IDX(i - 1, j - 1, m)] + cost) {
                D[IDX(i, j, m)] = D[IDX(i - 1, j - 1, m)] + cost;
                back[IDX(i, j, m)] = 0;
            }
        }
    }

    int pi = n - 1, pj = m - 1;
    int count = 0;
    while (pi >= 0 && pj >= 0) {
        path[count++] = (Point){pi, pj};
        int move = back[IDX(pi, pj, m)];
        if (move == 0) { pi--; pj--; }
        else if (move == 1) { pi--; }
        else if (move == 2) { pj--; }
        else break;
    }

    free(D);
    free(back);
    return count;
}

int expand_from_coarse_path(Point* coarse_path, int path_len, int radius, int n, int m, Point* window) {
    int count = 0;
    for (int k = 0; k < path_len; k++) {
        int i_c = coarse_path[k].i;
        int j_c = coarse_path[k].j;
        for (int i = 2 * i_c - radius; i <= 2 * i_c + 1 + radius; i++) {
            for (int j = 2 * j_c - radius; j <= 2 * j_c + 1 + radius; j++) {
                if (i >= 0 && i < n && j >= 0 && j < m) {
                    window[count++] = (Point){i, j};
                }
            }
        }
    }
    return count;
}

double constrained_dtw(double* x, int n, double* y, int m, Point* window, int window_size) {
    double* D = (double*)malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) D[i] = INFINITY;

    for (int k = 0; k < window_size; k++) {
        int i = window[k].i;
        int j = window[k].j;
        double cost = fabs(x[i] - y[j]);
        double min_prev = INFINITY;
        if (i > 0) min_prev = fmin(min_prev, D[IDX(i - 1, j, m)]);
        if (j > 0) min_prev = fmin(min_prev, D[IDX(i, j - 1, m)]);
        if (i > 0 && j > 0) min_prev = fmin(min_prev, D[IDX(i - 1, j - 1, m)]);
        if (i == 0 && j == 0) min_prev = 0.0;
        D[IDX(i, j, m)] = cost + min_prev;
    }

    double result = D[IDX(n - 1, m - 1, m)];
    free(D);
    return result;
}

double fastdtw_recursive(double* x, int n, double* y, int m, int radius) {
    if (n < 10 || m < 10) {
        Point full[n * m];
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                full[count++] = (Point){i, j};
            }
        }
        return constrained_dtw(x, n, y, m, full, count);
    }

    int n2, m2;
    double* x_reduced = (double*)malloc((n / 2) * sizeof(double));
    double* y_reduced = (double*)malloc((m / 2) * sizeof(double));
    reduce_by_half(x, n, x_reduced, &n2);
    reduce_by_half(y, m, y_reduced, &m2);

    Point* coarse_path = (Point*)malloc(n2 * m2 * sizeof(Point));
    int coarse_path_len = compute_dtw_with_path(x_reduced, n2, y_reduced, m2, coarse_path);

    Point* window = (Point*)malloc(n * m * sizeof(Point));
    int win_size = expand_from_coarse_path(coarse_path, coarse_path_len, radius, n, m, window);

    double result = constrained_dtw(x, n, y, m, window, win_size);

    free(x_reduced); free(y_reduced);
    free(coarse_path); free(window);
    return result;
}

static PyObject* py_fastdtw(PyObject* self, PyObject* args) {
    PyObject *x_obj, *y_obj;
    int radius = 1;
    if (!PyArg_ParseTuple(args, "OO|i", &x_obj, &y_obj, &radius))
        return NULL;

    PyArrayObject *x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr) return NULL;

    int n = (int)PyArray_DIM(x_arr, 0);
    int m = (int)PyArray_DIM(y_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);

    double cost = fastdtw_recursive(x, n, y, m, radius);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);

    return Py_BuildValue("d", cost);
}

static PyObject* py_fastdtw_matrix(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    int radius = 1;
    if (!PyArg_ParseTuple(args, "O|i", &input_obj, &radius)) return NULL;

    PyArrayObject* input = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!input) return NULL;

    int N = (int)PyArray_DIM(input, 0);
    int T = (int)PyArray_DIM(input, 1);
    double* data = (double*)PyArray_DATA(input);

    npy_intp dims[2] = {N, N};
    PyArrayObject* dist_mat = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    double* dist = (double*)PyArray_DATA(dist_mat);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double* x = data + i * T;
            double* y = data + j * T;
            double cost = fastdtw_recursive(x, T, y, T, radius);
            dist[i * N + j] = cost;
            dist[j * N + i] = cost;
        }
    }

    Py_DECREF(input);
    return (PyObject*)dist_mat;
}

static PyMethodDef Methods[] = {
    {"fastdtw", py_fastdtw, METH_VARARGS, "Compute FastDTW cost between two sequences with optional radius"},
    {"fastdtw_matrix", py_fastdtw_matrix, METH_VARARGS, "Compute all-pairs FastDTW distances from 2D NumPy array"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fastdtwmodule", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit_fastdtwmodule(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
