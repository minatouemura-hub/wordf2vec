#ifndef DTW_H
#define DTW_H

double compute_dtw_total_cost(double* x, int n, double* y, int m);

double* compute_dtw_accumulated_cost(double* x, int n, double* y, int m, int* rows, int* cols, double* final_cost);

#endif