#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include "dtw.h"

// 単一のDTW距離のみ
static PyObject* py_compute_dtw(PyObject* self, PyObject* args) {
    PyObject *x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;

    PyArrayObject *x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr) return NULL;

    int n = (int)PyArray_DIM(x_arr, 0);
    int m = (int)PyArray_DIM(y_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);

    double cost = compute_dtw_total_cost(x, n, y, m);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);

    return Py_BuildValue("d", cost);
}

// DTW + 累積コスト行列を返す
static PyObject* py_compute_dtw_with_matrix(PyObject* self, PyObject* args) {
    PyObject *x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;

    PyArrayObject *x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr) return NULL;

    int n = (int)PyArray_DIM(x_arr, 0);
    int m = (int)PyArray_DIM(y_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);

    int rows, cols;
    double final_cost;
    double* D = compute_dtw_accumulated_cost(x, n, y, m, &rows, &cols, &final_cost);

    npy_intp dims[2] = {rows, cols};
    PyArrayObject* cost_matrix = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(cost_matrix), D, sizeof(double) * rows * cols);

    free(D);
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);

    return Py_BuildValue("(O,d)", cost_matrix, final_cost);
}

// 全ユーザー組み合わせの DTW を並列に計算（OpenMP）
static PyObject* py_compute_dtw_matrix(PyObject* self, PyObject* args) {
    PyObject* input_array_obj;

    if (!PyArg_ParseTuple(args, "O", &input_array_obj))
        return NULL;

    PyArrayObject* input_array = (PyArrayObject*)PyArray_FROM_OTF(input_array_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!input_array) return NULL;

    int n_users = (int)PyArray_DIM(input_array, 0);
    int series_len = (int)PyArray_DIM(input_array, 1);

    npy_double* data = (npy_double*)PyArray_DATA(input_array);

    npy_intp dims[2] = {n_users, n_users};
    PyArrayObject* result = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    npy_double* dist = (npy_double*)PyArray_DATA(result);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_users; i++) {
        for (int j = i + 1; j < n_users; j++) {
            double* seq1 = data + i * series_len;
            double* seq2 = data + j * series_len;
            double cost = compute_dtw_total_cost(seq1, series_len, seq2, series_len);
            dist[i * n_users + j] = cost;
            dist[j * n_users + i] = cost;
        }
    }

    Py_DECREF(input_array);
    return (PyObject*)result;
}

// 関数定義テーブル
static PyMethodDef Methods[] = {
    {"compute_dtw", py_compute_dtw, METH_VARARGS, "Compute DTW total cost between two sequences"},
    {"compute_dtw_with_matrix", py_compute_dtw_with_matrix, METH_VARARGS, "Compute DTW cost matrix and final cost"},
    {"compute_dtw_matrix", py_compute_dtw_matrix, METH_VARARGS, "Compute pairwise DTW distances using OpenMP"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "dtwmodule", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit_dtwmodule(void) {
    import_array();  // NumPyの初期化
    return PyModule_Create(&moduledef);
}