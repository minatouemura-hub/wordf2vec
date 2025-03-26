import numpy as np


def compute_dtw_accumulated_cost(x, y):
    n = len(x)
    m = len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    acc_cost = D[1:, 1:]
    return acc_cost, acc_cost[-1, -1]


def resample_sequence(seq, target_length):
    original_length = len(seq)
    if original_length == target_length:
        return seq
    new_indices = np.linspace(0, original_length - 1, target_length)
    return np.interp(new_indices, np.arange(original_length), seq)
