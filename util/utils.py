from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def plt_linear(labels, data_gen: dict, output_path: Path):
    n_clusters = len(np.unique(labels))
    n_rows = int(np.ceil(n_clusters / 2))

    # === グラフ全体を2段構成（上段: 折れ線, 下段: ボックスプロット） ===
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(n_rows + 1, 2, height_ratios=[1] * n_rows + [0.5])

    dominat_freqs = []

    # --- クラスタごとの折れ線プロット ---
    for cl in range(n_clusters):
        row, col = divmod(cl, 2)
        ax = fig.add_subplot(gs[row, col])
        cluster_indices = np.where(labels == cl)[0]
        user_ids = list(data_gen.keys())
        # cluster_data: list of arrays
        cluster_data = [data_gen[user_ids[i]] for i in cluster_indices]

        # 各系列の長さを調べて最小長さに合わせて整形する（例）
        min_len = min(len(series) for series in cluster_data)
        cluster_data_trimmed = np.array([series[:min_len] for series in cluster_data])

        # 周波数分解（関数は別定義されている想定）
        dominat_freqs = foulier_decomp(cl, cl_data=cluster_data, dominant_freqs=dominat_freqs)
        dom_freq = np.median([f for c, f in dominat_freqs if c == cl])
        if dom_freq > 0:
            period = int(1 / dom_freq)
            n_cycles = 1
            x_max = min(cluster_data.shape[1], n_cycles * period)
        else:
            x_max = cluster_data.shape[1]
        # 各系列とその平均
        padded_data = []
        for series in cluster_data:
            trimmed = series[:x_max]
            ax.plot(trimmed, color="black", alpha=0.1)
            padded = np.pad(trimmed, (0, x_max - len(trimmed)), constant_values=np.nan)
            padded_data.append(padded)
        padded_data = np.array(padded_data)

        cluster_center = np.nanmean(padded_data, axis=0)
        ax.plot(cluster_center, color="red", linewidth=2)

        if dom_freq > 0:
            tick_positions = np.arange(0, x_max + 1, period)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{i+1}周期" for i in range(len(tick_positions))])
        ax.set_xlim(0, x_max)
        ax.set_ylabel("projection value")
        ax.set_title(f"Cluster {cl}")

    # --- 余った subplot 埋める ---
    for i in range(n_clusters, n_rows * 2):
        fig.add_subplot(gs[i // 2, i % 2]).axis("off")

    # --- ボックスプロット（主周波数分布）---
    df_freqs = pd.DataFrame(dominat_freqs, columns=["cluster", "dominant_freq"])
    ax_box = fig.add_subplot(gs[-1, :])
    sns.boxplot(data=df_freqs, x="cluster", y="dominant_freq", ax=ax_box)
    ax_box.set_title("Dominant Frequencies per Cluster")

    # --- 全体調整 & 保存 ---
    fig.suptitle("DTW distance K-means Clustering + Frequency Analysis", fontsize=16)
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
