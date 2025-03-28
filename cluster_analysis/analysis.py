import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "dtw_module"))
import gc  # noqa

import fastdtwmodule
import japanize_matplotlib  # noqa
import matplotlib.colors as mcolors  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import scipy.cluster.hierarchy as sch  # noqa
import seaborn as sns  # noqa
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage  # noqa
from scipy.spatial.distance import euclidean  # noqa
from scipy.spatial.distance import squareform  # noqa
from scipy.stats import gaussian_kde  # noqa
from sklearn.cluster import KMeans  # noqa
from sklearn.manifold import TSNE  # noqa
from tqdm import tqdm  # noqa
from tslearn.clustering import KernelKMeans, TimeSeriesKMeans  # noqa

from gender_axis.projection import Project_On  # noqa
from util.utils import foulier_decomp, plt_linear, resample_sequence  # noqa
from word2vec.train import Trainer  # noqa

# 必要なライブラリのインポート


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
        if not os.path.isfile(result_path):
            self.projection()
        self.projection_result = self._read_projection_result(result_path)
        self.base_dir = base_dir

    # 1.K-Meansによるクラスタリングとt-SNEによるサンプリングデータの可視化
    def k_means_tsne_plt(self):
        kmeans = KMeans(n_clusters=self.num_cluster, n_init="auto")
        self.cluster_label = kmeans.fit_predict(self.book_embeddings)

        # sampling
        indices = np.random.choice(
            self.book_embeddings.shape[0], size=self.num_samples, replace=False
        )
        sample_embeddings = self.book_embeddings[indices]
        sample_labels = self.cluster_label[indices]

        # 高次元の埋め込み表現（例：self.book_embeddings）を2次元に削減
        tsne = TSNE(n_components=2, random_state=42, perplexity=60)
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
        FIG_PATH = self.base_dir / "plt" / "t-SNE_fig.png"
        plt.savefig(FIG_PATH)
        # plt.show()

    def cluster_distribution(self):
        # 2.1 K-means のクラスタラベル + 書籍タイトルの DataFrame
        df_cluster = pd.DataFrame(
            {
                "book_title": [self.id2book[i] for i in range(len(self.book_embeddings))],
                "cluster_label": self.cluster_label,
            }
        )
        # 2.2 projection_result と結合してスカラー値を取得
        dist_df = pd.merge(self.projection_result, df_cluster, on="book_title", how="inner")

        dist_df["z_score"] = (
            dist_df["projection_vec"] - dist_df["projection_vec"].mean()
        ) / dist_df["projection_vec"].std()
        dist_df["percentile"] = dist_df["projection_vec"].rank(pct=True) * 100

        # ---(ポイント) カラーマップを作成: 赤→白→青 ---
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "my_cmap",
            [(0.0, "red"), (0.25, "orange"), (0.50, "lightgray"), (0.75, "cyan"), (1.0, "blue")],
        )

        # ---(ポイント) Seaborn の FacetGrid + 独自の描画関数を map ---
        g = sns.FacetGrid(
            dist_df,
            row="cluster_label",
            sharex=False,
            height=2.0,
            aspect=3.0,
            margin_titles=True,
        )

        # (A) 独自のKDE + fill_between (グラデーション) 関数
        def gradient_kdeplot(x, *args, **kwargs):
            """
            FacetGrid.map() から呼ばれ、同一クラスタのデータ x (percentile列) を受け取る。
            x の分布をKDEで推定し、左右方向で色が変わるグラデーション塗りつぶしを行う。
            """
            ax = plt.gca()

            # 1) カーネル密度推定を計算
            kde = gaussian_kde(x)
            # Seabornのbw_adjust=0.8 相当のバンド幅調整(任意で変更可)
            bw_adjust = 0.8
            kde.set_bandwidth(kde.factor * bw_adjust)

            # 2) 0～100 を細かく分割して KDE を評価
            X = np.linspace(0, 100, 500)
            Y = kde.evaluate(X)

            # 3) fill_between で小区間ごとに塗りつぶす
            for i in range(len(X) - 1):
                # 区間の中央 x 値
                x_mean = 0.5 * (X[i] + X[i + 1])
                # 0～100 の範囲を [0,1] に正規化して cmap に渡す
                t = x_mean / 100.0
                color = cmap(t)

                ax.fill_between(
                    X[i : i + 2],  # noqa
                    Y[i : i + 2],  # noqa
                    color=color,
                    interpolate=True,
                )

        # (B) FacetGrid に対して、上記関数を適用
        g.map(gradient_kdeplot, "percentile")

        # ---(ポイント) 軸やスパインなどの装飾を調整 ---
        for ax in g.axes.flatten():
            # x範囲を 0~100 に固定
            ax.set_xlim(0, 100)

            # y軸目盛りは不要
            ax.set_yticks([])

            # x軸目盛りを消し、数値ラベルも表示しない
            ax.set_xticks([])

            # スパイン（枠線）の可視化を制御
            ax.spines["bottom"].set_visible(True)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # 中央線 x=50 の破線
            # ax.axvline(x=50, color="b", linestyle="--", alpha=0.8)

        plt.subplots_adjust(left=0.15, top=0.9, hspace=0.05)
        g.set_titles("")
        # 画像ファイルとして保存
        fig_path = self.base_dir / "plt" / "cluster_distribution.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        # plt.show()

        # 各クラスタ内の書籍タイトル例を表示
        for cl in range(self.num_cluster):
            titles_in_cl = dist_df.loc[dist_df["cluster_label"] == cl, "book_title"]
            print(f"Cluster {cl} sample titles:", titles_in_cl.head(5).tolist())

    def make_correlogram_from_dict(self):
        """
        読書履歴が最も長いユーザー、最も短いユーザー、平均的なユーザー各2名について、
        読書順に沿った projection_vec の自己相関（コレログラム）を作成します。

        Parameters:
            history_dict (dict):
                { user_id: [ { "book_title": ..., "author": ... }, ... ] }
                ※ 各ユーザーのリストは読んだ順番になっていると仮定。
        """
        # 各ユーザーの読書数をカウント
        user_counts = {user: len(self.data_gen[user]) for user in self.data_gen}
        user_counts_df = pd.DataFrame(list(user_counts.items()), columns=["user_id", "count"])
        avg_count = user_counts_df["count"].mean()

        # 読書数が最も多いユーザー上位2名、最も少ないユーザー下位2名、
        # および平均に近いユーザー（平均との差が小さい上位2名）を選択
        longest_users = (
            user_counts_df.sort_values("count", ascending=False).head(2)["user_id"].tolist()
        )
        # shortest_users = (
        #     user_counts_df.sort_values("count", ascending=True).head(2)["user_id"].tolist()
        # )
        user_counts_df["diff"] = (user_counts_df["count"] - avg_count).abs()
        average_users = user_counts_df.sort_values("diff").head(2)["user_id"].tolist()

        selected_users = {
            "Longest": longest_users,
            # "Shortest": shortest_users,
            "Average": average_users,
        }

        # 3行2列のプロット領域を作成
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        fig.suptitle("Correlograms of Selected Users' Reading Histories", fontsize=16)

        for row, (category, users) in enumerate(selected_users.items()):
            for col, user in enumerate(users):
                # 辞書から対象ユーザーの読書履歴リストを DataFrame 化
                user_history = pd.DataFrame(self.data_gen[user])
                # 読書順はリストの順番なので、その順序を保持するためのインデックスを付与
                user_history["order"] = range(len(user_history))

                user_history.rename(columns={"Title": "book_title"}, inplace=True)
                # projection_result（book_title と projection_vec）とマージ
                merged = pd.merge(
                    user_history, self.projection_result, on="book_title", how="inner"
                )
                # 元の読書順(order)でソート
                merged = merged.sort_values("order")

                # projection_vec を数値型に変換
                try:
                    series = merged["projection_vec"].astype(float)
                except Exception as e:
                    print(f"User {user} の projection_vec の変換に失敗: {e}")
                    continue

                # ラグ1～min(20, len(series)-1) の自己相関を計算
                max_lag = min(20, len(series) - 1)
                lags = range(1, max_lag + 1)
                acf_values = [series.autocorr(lag=lag) for lag in lags]

                ax = axes[row, col]
                ax.bar(lags, acf_values)
                ax.set_title(f"User {user} ({category})")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Autocorrelation")
                ax.set_ylim(-1, 1)
                ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = self.base_dir / "plt" / "user_correlograms_from_dict.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        # plt.show()

    def dtw_kmeans_users(self, max_clusters=5, plt_flag=True, gamma=0.01):
        """
        各ユーザーの読書順に沿った projection_vec を時系列データとみなし、
        DTW距離を用いて TimeSeriesKMeans によるクラスタリングを実施します。

        各ユーザーの読書履歴の長さが異なるため、線形補間により固定長にリサンプリングしています。
        また、オプションで各クラスタの全ユーザー間の累積コスト行列をDTWで計算し、
        平均したヒートマップを表示します。（ヒートマップのカラースケールは全クラスタで共通）
        """

        # --- (1) ユーザーごとの時系列データを抽出 ---
        user_series = {}
        for user in self.data_gen:
            user_history = pd.DataFrame(self.data_gen[user])
            if len(user_history) == 0:
                continue
            user_history["order"] = range(len(user_history))
            user_history.rename(columns={"Title": "book_title"}, inplace=True)
            merged = pd.merge(user_history, self.projection_result, on="book_title", how="inner")
            merged = merged.sort_values("order")
            try:
                series = merged["projection_vec"].astype(float).values
            except Exception as e:
                print(f"User {user} の projection_vec の変換に失敗: {e}")
                continue
            if len(series) > 1:
                user_series[user] = series

        user_ids = list(user_series.keys())
        n_users = len(user_ids)
        if n_users < 2:
            print("クラスタリングに十分なユーザーが存在しません。")
            return
        # --- (3) DTWと累積コスト行列の計算---
        gc.collect()
        dtw_distances = np.zeros((n_users, n_users), dtype=np.float32)
        for i in tqdm(range(n_users), desc="DTW Processing...", leave=True):
            for j in tqdm(
                range(i + 1, n_users), desc=f"User_Number_{i} Processing...", leave=False
            ):
                seq1 = user_series[user_ids[i]]
                seq2 = user_series[user_ids[j]]

                dist = fastdtwmodule.fastdtw(seq1, seq2)

                dtw_distances[i, j] = dist
                dtw_distances[j, i] = dist
        print("==DTW Computed==")
        # --- (4) クラスタリング (KernelKMeans: 事前計算済みDTW距離からカーネル行列を構築) ---
        dtw_kernel = np.exp(-gamma * (dtw_distances**2))
        kernel_km = KernelKMeans(n_clusters=5, kernel="precomputed", random_state=42)

        cluster_labels = kernel_km.fit_predict(dtw_kernel)

        self.plot_relative_gender_ratio_by_cluster(cluster_labels=cluster_labels, user_ids=user_ids)
        plt_linear(labels=cluster_labels, data_gen=user_series, output_path=self.base_dir / "plt")

    def plot_relative_gender_ratio_by_cluster(self, cluster_labels, user_ids):
        """
        各性別に対して「このクラスタでは相対的に多いか？」を示すヒートマップを表示

        Parameters:
            cluster_labels (np.array): 各ユーザーのクラスタラベル
            user_ids (List[str]): user_seriesのキー（ユーザーID）リストと一致
        """
        gender_data = []

        for idx, user_id in enumerate(user_ids):
            gender = self.data_gen[user_id][0].get("Gender", "Unknown")
            cluster = cluster_labels[idx]
            gender_data.append({"cluster": cluster, "gender": gender})

        df = pd.DataFrame(gender_data)

        # クロス集計: クラスタ×性別 の人数テーブル
        crosstab = pd.crosstab(df["gender"], df["cluster"])

        # 性別ごとに z-score を計算（クラスタ内での偏りを相対評価）
        zscore_df = crosstab.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

        # ヒートマップ描画
        plt.figure(figsize=(10, 5))
        sns.heatmap(zscore_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f")

        plt.title("Relative Gender Ratio by Cluster (Z-score)")
        plt.xlabel("Cluster")
        plt.ylabel("Gender")
        plt.tight_layout()

        fig_path = self.base_dir / "plt" / "gender_ratio_relative_zscore.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300)
        plt.close()
