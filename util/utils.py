from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tslearn.barycenters import dtw_barycenter_averaging


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


def plt_linear(labels, data_gen: dict, output_path: Path, target_len: int = 30):
    # --- 対象ユーザーを target_len 以上にフィルタ ---
    valid_user_ids = [uid for uid, seq in data_gen.items() if len(seq) >= target_len]

    # 対象ユーザーのインデックスとラベルだけ残す
    user_id_list = list(data_gen.keys())
    user_id_to_index = {uid: i for i, uid in enumerate(user_id_list)}
    valid_indices = [user_id_to_index[uid] for uid in valid_user_ids]

    labels = np.array(labels)
    filtered_labels = labels[valid_indices]

    # フィルタ後の data_gen を作成
    filtered_data_gen = {uid: data_gen[uid] for uid in valid_user_ids}

    n_clusters = len(np.unique(filtered_labels))
    n_rows = int(np.ceil(n_clusters / 2))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(n_rows + 1, 2, height_ratios=[1] * n_rows + [0.5])

    for cl in range(n_clusters):
        row, col = divmod(cl, 2)
        ax = fig.add_subplot(gs[row, col])
        cluster_indices = np.where(filtered_labels == cl)[0]
        filtered_user_ids = list(filtered_data_gen.keys())
        cluster_data = [filtered_data_gen[filtered_user_ids[i]] for i in cluster_indices]

        # リサンプルして長さを統一
        cluster_data_resampled = np.array(
            [resample_sequence(series, target_len) for series in cluster_data]
        )

        # 各系列を描画
        for series in cluster_data_resampled:
            ax.plot(series, color="black", alpha=0.1)

        # DTWバリセンターをクラスタ中心とする
        cluster_center = dtw_barycenter_averaging(cluster_data_resampled)
        ax.plot(cluster_center, color="red", linewidth=2)

        ax.set_xlim(0, target_len)
        ax.set_ylabel("projection value")
        ax.set_title(f"Cluster {cl}")

    # --- 余った subplot 埋める ---
    for i in range(n_clusters, n_rows * 2):
        fig.add_subplot(gs[i // 2, i % 2]).axis("off")

    fig.suptitle("DTW distance K-means Clustering + DTW Barycenter Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "dtw_cluster_plot.png")
    plt.close()


def foulier_decomp(
    cl: int,
    cl_data: np.array,
    dominant_freqs: list,
):
    for series in cl_data:
        fft_vals = np.fft.fft(series - np.mean(series))
        freqs = np.fft.fftfreq(len(series))
        mag = np.abs(fft_vals[1 : len(freqs) // 2])
        if mag.size == 0:
            continue
        dominant = freqs[1 : len(freqs) // 2][np.argmax(mag)]  # noqa
        dominant_freqs.append((cl, dominant))
    return dominant_freqs
