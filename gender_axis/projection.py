import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from word2vec.train import Trainer


class Project_On(Trainer):
    def __init__(
        self,
        base_dir: Path,
        folder_path: Path,
        weight_path: Path,
        top_k=10,
        embedding_dim=50,
        num_negatives=5,
        batch_size=32,
        epochs=5,
        learning_rate=0.005,
    ):
        super().__init__(
            folder_path,
            weight_path,
            embedding_dim,
            num_negatives,
            batch_size,
            epochs,
            learning_rate,
        )
        self.base_dir = base_dir
        self.top_k = top_k
        if os.path.isfile(weight_path):
            self._read_weight_vec(device="cpu")
            self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        else:
            self.run_train()

    def projection(self, pca_flag=True):
        print("\n==射影を開始します==")
        social_axis = self.find_gender_axis(self.top_k)
        normed_social_axis = social_axis / np.linalg.norm(social_axis)
        if pca_flag:
            self._principal_components_plt()
        # 進捗管理したいので計算速度落ちてもって感じ...
        projections = np.empty((self.book_embeddings.shape[0],))
        for i in tqdm(range(self.book_embeddings.shape[0]), desc="Projecting embeddings"):
            # for i in tqdm(range(1, 100, 1), desc="Projecting embeddings"):
            projections[i] = np.dot(self.book_embeddings[i], normed_social_axis)
        self._save_projections(projections)
        return projections

    def _save_projections(self, projections):
        RESULT_PATH = self.base_dir / "result" / "projection_result.csv"
        with open(RESULT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["book_title", "projection_vec"])
            for idx, proj in enumerate(projections):
                book_title = self.id2book.get(idx, "Unknow")
                writer.writerow([book_title, proj])
        print(f"==射影の結果は{RESULT_PATH}に保存されています==")

    def _principal_components_plt(self):
        if self.selected_diff_vectors.ndim == 1:
            raise ValueError("社会軸の構成成分が1つしかありません")  # 2次元配列に変更
        self.pca = PCA()
        self.pca.fit(self.selected_diff_vectors)
        explained_variance = self.pca.explained_variance_ratio_
        pc_labels = [f"PC{i+1}" for i in range(len(explained_variance))]

        plt.figure(figsize=(8, 6))  # 図のサイズを指定（8×6インチ）
        sns.barplot(x=pc_labels, y=explained_variance, color="#636EFA", alpha=0.9)

        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA - Explained Variance Ratio of Principal Components")
        plt.show()

    def find_gender_axis(
        self,
        k: int = 10,  # 全体で使用するペア数（seed ペア含む）
        n_neighbors: int = 10,  # 各書籍について取得する近傍数（自身を含むので実質10近傍）
        male_book: str = "できる男の顔になるフェイス・ビルダー: 人生を変えるフェイシャル筋トレ",
        female_book: str = "つけるだけでお腹が凹む! ダイエット背中ブラ【背中ブラ付き】 (TJMOOK)",
    ):
        # 1. seed ペアの取得
        male_fav_bookid = self.book2id[male_book]
        female_fav_bookid = self.book2id[female_book]
        # seed ペアは (male, female) の順に固定
        seed_pair = (male_fav_bookid, female_fav_bookid)
        seed_diff = self._compute_diff_vec(seed_pair).reshape(1, -1)

        # 2. 各書籍について、近傍 (10近傍) を取得して候補ペアリストを作成

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(self.book_embeddings)
        distances, indices = nbrs.kneighbors(self.book_embeddings)

        candidate_pairs = []
        N = len(self.book_embeddings)
        # 各書籍 i について、自身（最初の要素）を除く近傍を候補として (i, neighbor) ペアを追加
        for i in range(N):
            for neighbor in indices[i, 1:]:
                candidate_pairs.append((i, int(neighbor)))
        candidate_pairs = np.array(candidate_pairs)  # shape: (num_candidates, 2)

        # 3. seed ペアを候補から除外（seed ペアは後で必ず先頭に追加）
        mask = ~(
            (candidate_pairs[:, 0] == male_fav_bookid)
            & (candidate_pairs[:, 1] == female_fav_bookid)
        )
        candidate_pairs = candidate_pairs[mask]

        # 4. 各候補ペアの差分ベクトルを一括計算し、seed 差分との cosine 類似度を算出
        diff_vectors = (
            self.book_embeddings[candidate_pairs[:, 0]]
            - self.book_embeddings[candidate_pairs[:, 1]]
        )
        sims = cosine_similarity(seed_diff, diff_vectors)[0]  # shape: (num_candidates,)

        # 5. 候補ペアを (pair, sim) のタプルとしてまとめ、類似度降順にソート
        candidate_list = [(tuple(pair), sim_val) for pair, sim_val in zip(candidate_pairs, sims)]
        candidate_list.sort(key=lambda x: x[1], reverse=True)

        # 6. グリーディにペアを選択（seed ペアのコミュニティを含むものは除外）
        selected_pairs = [(seed_pair, 1.0)]  # seed ペアの類似度は 1.0 とする
        selected_set = set(seed_pair)
        for pair, sim_val in candidate_list:
            # 既に選ばれたコミュニティに含まれていなければ選択
            if (pair[0] not in selected_set) and (pair[1] not in selected_set):
                selected_pairs.append((pair, sim_val))
                selected_set.update(pair)
            if len(selected_pairs) >= k:
                break

        # 7. 結果表示（各ペアと cosine 類似度）
        for pair, sim_val in selected_pairs:
            title1 = self.id2book.get(pair[0], "Unknown")
            title2 = self.id2book.get(pair[1], "Unknown")
            print(f"({title1}, {title2}): sim={sim_val:.4f}")

        # 8. 選択された各ペアの差分ベクトルを計算し平均して社会軸を求める
        self.selected_diff_vectors = np.array(
            [self._compute_diff_vec(pair) for pair, _ in selected_pairs]
        )
        social_dimension = self.selected_diff_vectors.mean(axis=0)
        return social_dimension

    def _compute_diff_vec(self, pair):
        s1, s2 = pair
        return self.book_embeddings[s1] - self.book_embeddings[s2]

    def _read_projection_result(self, result_path: Path):
        return pd.read_csv(result_path)
