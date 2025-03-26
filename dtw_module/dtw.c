#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "dtw.h"

#define IDX(i, j, cols) ((i) * (cols) + (j))

double compute_dtw_total_cost(double* x, int n, double* y, int m) {
    int i, j;
    int rows = n + 1, cols = m + 1;
    double* D = (double*)malloc(rows * cols * sizeof(double));
    for (i = 0; i < rows * cols; i++) D[i] = INFINITY;
    D[IDX(0, 0, cols)] = 0.0;

    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            double cost = fabs(x[i - 1] - y[j - 1]);
            double min_prev = fmin(D[IDX(i - 1, j, cols)],
                                   fmin(D[IDX(i, j - 1, cols)], D[IDX(i - 1, j - 1, cols)]));
            D[IDX(i, j, cols)] = cost + min_prev;
        }
    }

    double result = D[IDX(n, m, cols)];
    free(D);
    return result;
}

double* compute_dtw_accumulated_cost(double* x, int n, double* y, int m, int* rows, int* cols, double* final_cost) {
    int i, j;
    int r = n + 1, c = m + 1;

    double* D = (double*)malloc(r * c * sizeof(double));
    for (i = 0; i < r * c; i++) D[i] = INFINITY;
    D[IDX(0, 0, c)] = 0.0;

    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            double cost = fabs(x[i - 1] - y[j - 1]);
            double min_prev = fmin(D[IDX(i - 1, j, c)],
                                   fmin(D[IDX(i, j - 1, c)], D[IDX(i - 1, j - 1, c)]));
            D[IDX(i, j, c)] = cost + min_prev;
        }
    }

    *final_cost = D[IDX(n, m, c)];

    // --- ここから: D[1:,1:] をコピーして返す ---
    double* D_trimmed = (double*)malloc(n * m * sizeof(double));
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            D_trimmed[i * m + j] = D[IDX(i + 1, j + 1, c)];
        }
    }

    *rows = n;
    *cols = m;

    free(D);  // 元のDを解放
    return D_trimmed;
}