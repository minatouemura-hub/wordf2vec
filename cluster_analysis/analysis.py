from math import ceil  # noqa
from pathlib import Path
from typing import Tuple

import japanize_matplotlib  # noqa
import matplotlib.colors as mcolors  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import ruptures as rpt  # noqa
import scipy.cluster.hierarchy as sch  # noqa
import scipy.stats as stats  # noqa
import seaborn as sns  # noqa
from kneed import KneeLocator  # noqa
from scipy.stats import gaussian_kde  # noqa
from sklearn.cluster import KMeans  # noqa
from sklearn.manifold import MDS  # noqa
from sklearn.manifold import TSNE  # noqa
from sklearn.metrics import label_ranking_average_precision_score  # noqa
from sklearn.preprocessing import MultiLabelBinarizer  # noqa
from torch import nn  # noqa
from tqdm import tqdm  # noqa
from tslearn.clustering import KernelKMeans, TimeSeriesKMeans  # noqa

from arg import ClusterConfig
from gender_axis.projection import Project_On  # noqa
from word2vec import Trainer  # noqa


# 必要なライブラリのインポート
class Balanced_Kmeans(ClusterConfig):
    def __init__(self, vec_df: pd.DataFrame, meta_df: pd.DataFrame, n_cluster: int = 5):
        self.vec_df = vec_df
        self.meta_df = meta_df
        self.n_clusters = n_cluster

    def balanced_kmeans(self, X, n_clusters):
        """
        各クラスタのサイズに対して柔軟な制約を持つk-means（バランス緩和版）

        Parameters:
        tolerance_ratio: float
            平均クラスタサイズに対する許容比率（例: 0.1 → ±10%）
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        avg = n_samples / n_clusters
        min_cap = int(avg * (1 - self.tolerance_ratio))  # noqa
        max_cap = int(avg * (1 + self.tolerance_ratio))

        initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[initial_indices].copy()
        labels = np.zeros(n_samples, dtype=int)

        for it in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
            sorted_idx = np.argsort(distances, axis=1)

            new_labels = -np.ones(n_samples, dtype=int)
            cluster_sizes = {i: 0 for i in range(n_clusters)}
            order = np.argsort(np.min(distances, axis=1))

            for i in order:
                for c in sorted_idx[i]:
                    if cluster_sizes[c] < max_cap:
                        new_labels[i] = c
                        cluster_sizes[c] += 1
                        break

            # 未割当があれば強制割当（最大容量無視）
            unassigned = np.where(new_labels == -1)[0]
            for i in unassigned:
                for c in sorted_idx[i]:
                    new_labels[i] = c
                    cluster_sizes[c] += 1
                    break

            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                points = X[new_labels == c]
                new_centroids[c] = points.mean(axis=0) if len(points) > 0 else centroids[c]

            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            if shift < self.tol:
                break
            centroids = new_centroids
            labels = new_labels
        return labels, centroids

    def compute_inertia(self, X, labels, centroids):
        """
        inertia = sum( || x_i - centroid(label_i) ||^2 ) を計算
        """
        inertia = 0.0
        for c in range(centroids.shape[0]):
            points = X[labels == c]
            if len(points) > 0:
                inertia += np.sum((points - centroids[c]) ** 2)
        return inertia

    def evaluate_clustering_with_genre_sets(
        self,
        id_col: str = "title",
        genre_col: str = "genres",
        save_dir: Path = Path("."),
    ):
        """
        複数ジャンルを持つデータに対して、クラスタリング結果との一致度をマルチラベル評価する。
        さらに Elbow 法によるクラスタ数のチューニング（バランス付き k-means を使用）を実施し、
        最適なクラスタ数でクラスタリングを行う。

        Parameters:
        vec_df: ベクトル表現（index=title）
        meta_df: ジャンル列（genres列）が含まれるメタ情報
        id_col: vec_df.index に対応するカラム名（例：title）
        genre_col: 'Action|Adventure|Sci-Fi' などの複数ジャンル列
        n_clusters: （初期値）クラスタ数。Elbow 法で再推定される。
        save_dir: 結果の保存先

        Returns:
        cluster_genre_dist: 各クラスタのジャンル頻度（絶対値）
        lrap: Label Ranking Average Precision
        cluster_labels: 最適なクラスタ数によるクラスタリングのラベル配列
        optimal_k: 推定された最適クラスタ数
        """
        print("\n▶️ マルチジャンルによるクラスタ評価を実行中...")

        X = self.vec_df.values
        inertia_list = []
        cluster_range = list(range(15, 51))
        for k in cluster_range:
            # balanced_kmeans によりクラスタリング
            labels_temp, centroids_temp = self.balanced_kmeans(X, k)
            inertia = self.compute_inertia(X, labels_temp, centroids_temp)
            inertia_list.append(inertia)

        kl = KneeLocator(cluster_range, inertia_list, curve="convex", direction="decreasing")
        optimal_k = kl.knee if kl.knee is not None else self.n_clusters
        print(f"推定された最適クラスタ数: {optimal_k}")

        # 最適なクラスタ数を用いてバランス付き k-means を実行
        cluster_labels, centroids = self.balanced_kmeans(X, optimal_k)

        # 統合処理
        df = self.vec_df.reset_index().copy()
        df["cluster"] = cluster_labels
        df = df.merge(self.meta_df[[id_col, genre_col]], on=id_col)

        # ジャンルリストに変換
        df["genre_list"] = df[genre_col].apply(lambda x: x.split("|") if pd.notnull(x) else [])

        # マルチラベルバイナリ

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(df["genre_list"])
        genres = mlb.classes_

        # クラスタごとのジャンル頻度の集計
        cluster_genre_dist = pd.DataFrame(0, index=range(optimal_k), columns=genres)
        for cl in range(optimal_k):
            labels_list = df[df["cluster"] == cl]["genre_list"]
            counts = pd.Series([g for sub in labels_list for g in sub]).value_counts()
            for g in counts.index:
                cluster_genre_dist.loc[cl, g] = counts[g]

        # 正規化（確率）
        cluster_genre_dist_norm = cluster_genre_dist.div(cluster_genre_dist.sum(axis=1), axis=0)

        plt.figure(figsize=(14, 6))
        zscore_df = (
            cluster_genre_dist_norm - cluster_genre_dist_norm.mean()
        ) / cluster_genre_dist_norm.std()
        sns.heatmap(zscore_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title("Cluster-wise Multi-Genre Distribution (z-score)")
        plt.ylabel("Cluster")
        plt.xlabel("Genre")
        plt.tight_layout()
        plt.savefig(save_dir / "plt" / "cluster_genre_zscore.png")
        plt.close()

        genre_scores = df["cluster"].map(cluster_genre_dist_norm.to_dict(orient="index"))
        genre_score_matrix = (
            pd.DataFrame(list(genre_scores), index=df.index)[mlb.classes_].fillna(0).values
        )

        lrap = label_ranking_average_precision_score(Y, genre_score_matrix)
        print(f"✅ Label Ranking Average Precision (LRAP): {lrap:.4f}")

        return cluster_genre_dist, lrap, cluster_labels


def compare_cluster_entropy_by_gender(data_df, cluster_labels, id2book, save_dir, num_cluster):
    """
    各ユーザーのクラスタ分布のエントロピーを計算し、性別ごとに比較
    """
    print("▶️ 性別ごとのクラスタ分布エントロピーを比較中...")

    # 書籍 → クラスタ辞書
    book2cluster = {id2book[i]: cluster_labels[i] for i in range(len(cluster_labels))}

    # ユーザーごとにクラスタを記録
    entropies = []
    for uid, group in data_df.groupby("userId"):
        gender = group["gender"].iloc[0] if "gender" in group.columns else "Unknown"
        titles = group["title"].dropna()
        cluster_list = titles.map(book2cluster).dropna().astype(int).tolist()
        if len(cluster_list) == 0:
            continue
        # クラスタ出現頻度
        counts = pd.Series(cluster_list).value_counts().reindex(range(num_cluster), fill_value=0)
        probs = counts / counts.sum()
        entropy = stats.entropy(probs, base=2)  # 情報エントロピー
        entropies.append({"userId": uid, "gender": gender, "entropy": entropy})

    df_entropy = pd.DataFrame(entropies)

    # ヒストグラム or ボックスプロットで表示
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_entropy, x="gender", y="entropy")
    plt.title("Cluster Distribution Entropy by Gender")
    plt.ylabel("Entropy (bits)")
    plt.xlabel("Gender")
    plt.tight_layout()

    fig_path = save_dir / "plt" / "gender_cluster_entropy.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 統計比較も（オプション）
    try:
        m_vals = df_entropy[df_entropy["gender"] == "M"]["entropy"]
        f_vals = df_entropy[df_entropy["gender"] == "F"]["entropy"]
        stat, p = stats.ttest_ind(m_vals, f_vals, equal_var=False)
        print(
            f"✅ 平均エントロピー 男性={m_vals.mean():.3f}, 女性={f_vals.mean():.3f}, p値={p:.4f}"
        )
    except Exception:
        print("⚠️ t検定失敗（データ不足の可能性）")


def detect_user_change_points(vec_series: np.ndarray, pen: int = 5) -> list:
    """vec_series: shape (T, D)"""
    if vec_series.shape[0] < 3:
        return []  # 長さが短すぎて変化点検出不能
    model = rpt.Pelt(model="rbf").fit(vec_series)
    return model.predict(pen=pen)[:-1]  # 最後の点は除く


def find_best_k_by_elbow(user_embeddings, max_k=10):
    distortions = []
    K = range(5, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_embeddings)
        distortions.append(kmeans.inertia_)

    deltas = [distortions[i - 1] - distortions[i] for i in range(1, len(distortions))]
    best_k = K[deltas.index(max(deltas))]
    return best_k


def evaluate_clustering_with_genre_sets(
    vec_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    id_col: str = "title",
    genre_col: str = "genres",
    save_dir: Path = Path("."),
    k_range: range = range(15, 51),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, float, np.ndarray]:
    from sklearn.cluster import KMeans

    print("\n▶️ マルチジャンルによるクラスタ評価を実行中...")

    X = vec_df.values
    inertia_list = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertia_list.append(kmeans.inertia_)

    kl = KneeLocator(list(k_range), inertia_list, curve="convex", direction="decreasing")
    optimal_k = kl.knee if kl.knee is not None else k_range.start
    print(f"推定された最適クラスタ数: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)

    df = vec_df.reset_index().copy()
    df["cluster"] = cluster_labels
    df = df.merge(meta_df[[id_col, genre_col]], on=id_col)
    df["genre_list"] = df[genre_col].apply(lambda x: x.split("|") if pd.notnull(x) else [])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["genre_list"])
    genres = mlb.classes_

    cluster_genre_dist = pd.DataFrame(0, index=range(optimal_k), columns=genres)
    for cl in range(optimal_k):
        labels_list = df[df["cluster"] == cl]["genre_list"]
        counts = pd.Series([g for sub in labels_list for g in sub]).value_counts()
        for g in counts.index:
            cluster_genre_dist.loc[cl, g] = counts[g]

    cluster_genre_dist_norm = cluster_genre_dist.div(cluster_genre_dist.sum(axis=1), axis=0)

    plt.figure(figsize=(14, 6))
    zscore_df = (
        cluster_genre_dist_norm - cluster_genre_dist_norm.mean()
    ) / cluster_genre_dist_norm.std()
    sns.heatmap(zscore_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Cluster-wise Multi-Genre Distribution (z-score)")
    plt.ylabel("Cluster")
    plt.xlabel("Genre")
    plt.tight_layout()
    plt.savefig(save_dir / "plt" / "cluster_genre_zscore.png")
    plt.close()

    genre_scores = df["cluster"].map(cluster_genre_dist_norm.to_dict(orient="index"))
    genre_score_matrix = (
        pd.DataFrame(list(genre_scores), index=df.index)[mlb.classes_].fillna(0).values
    )

    lrap = label_ranking_average_precision_score(Y, genre_score_matrix)
    print(f"✅ Label Ranking Average Precision (LRAP): {lrap:.4f}")

    return cluster_genre_dist, lrap, cluster_labels
