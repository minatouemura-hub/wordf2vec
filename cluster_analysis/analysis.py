import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch  # noqa
from scipy.cluster.hierarchy import fcluster  # noqa
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from gender_axis.projection import Project_On
from word2vec.train import Trainer  # noqa


class ClusterAnalysis(Project_On):
    def __init__(
        self,
        base_dir: Path,
        folder_path: Path,
        weight_path: Path,
        result_path: Path,
        num_cluster: int = 10,
        num_samples: int = 10000,
    ):
        Project_On.__init__(self, base_dir, folder_path, weight_path)
        self.num_cluster = num_cluster
        self.num_samples = num_samples
        if os.path.isfile(result_path):
            self.projeciton_result = self._read_projection_result(result_path=result_path)
        else:
            self.projeciton_result = self.projection()

    # 1.K-Meansによるクラスタリングとt-SNEによるサンプリングデータの可視化
    def k_means_tsne_plt(self):
        kmeans = KMeans(n_clusters=self.num_cluster, n_init="auto")
        cluster_label = kmeans.fit_predict(self.book_embeddings)

        # sampling
        indices = np.random.choice(
            self.book_embeddings.shape[0], size=self.num_samples, replace=False
        )
        sample_embeddings = self.book_embeddings[indices]
        sample_labels = cluster_label[indices]

        # 高次元の埋め込み表現（例：self.book_embeddings）を2次元に削減
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(sample_embeddings)

        # 2次元埋め込み表現に対して階層的クラスタリング（ウォード法）を適用
        # linkage_matrix = sch.linkage(embeddings_2d, method="ward")

        # --- k-means のクラスタラベルで可視化 ---
        plt.figure(figsize=(8, 6))
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=sample_labels,
            cmap="tab20",
            edgecolors="none",
            s=15,
            alpha=0.4,
        )
        plt.axis("off")

        plt.show()


# 2. 上記のクラスタリングの結果を用いて分布を実装
