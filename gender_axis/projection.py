import argparse
import csv
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from word2vec.train import BookDataset, Trainer


class Project_On(Trainer):
    def __init__(
        self,
        base_dir: Path,
        folder_path: Path,
        weight_path: Path,
        whole_args,
        projection_args,
        word2vec_config,
        top_k=10,
    ):

        self.base_dir = base_dir
        self.top_k = top_k
        self.projection_args = projection_args
        down_sample = word2vec_config.down_sample
        sample = word2vec_config.sample
        dataset = BookDataset(folder_path, down_sample=down_sample, sample=sample)
        dataset.execute()

        if os.path.isfile(weight_path) and not whole_args.retrain:
            self._read_weight_vec(device="cpu")
            self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        else:
            if whole_args.grid_search_flag:
                max_acc = -1
                hyparams = []
                embedding_dims = word2vec_config.size_range
                num_negatives = word2vec_config.negative_range
                learning_rates = word2vec_config.alpha_range
                for dim, neg, lr in tqdm(
                    product(embedding_dims, num_negatives, learning_rates),
                    desc="Grid Search Processing",
                    total=2 * 2 * 2,
                ):
                    Trainer.__init__(
                        self,
                        folder_path,
                        weight_path,
                        dataset=dataset,
                        embedding_dim=dim,
                        num_negatives=neg,
                        learning_rate=lr,
                        batch_size=124,
                        epochs=1,
                    )
                    acc = self.run_train()
                    if acc > max_acc:
                        hyparams.append([dim, neg, lr])
                quit()
            else:
                Trainer.__init__(
                    self,
                    folder_path,
                    weight_path,
                    dataset=dataset,
                    embedding_dim=word2vec_config.embedding_dim,
                    num_negatives=word2vec_config.num_negatives,
                    learning_rate=word2vec_config.learning_rate,
                    batch_size=word2vec_config.batch_size,
                    epochs=word2vec_config.epochs,
                    early_stop_threshold=word2vec_config.early_stop_threshold,
                    top_range=word2vec_config.top_range,
                    eval_task=word2vec_config.task_name,
                )
                acc = self.run_train()

    def projection(self, pca_flag=True):
        print("\n==射影を開始します==")
        social_axis = self.find_gender_axis(self.top_k)
        self._eval_axis(social_axis)
        normed_social_axis = social_axis / np.linalg.norm(social_axis)
        if pca_flag:
            self._principal_components_plt()
        # 進捗管理したいので計算速度落ちてもって感じ...
        projections = np.empty((self.book_embeddings.shape[0],))
        for i in tqdm(range(self.book_embeddings.shape[0]), desc="Projecting embeddings"):
            # for i in tqdm(range(1, 100, 1), desc="Projecting embeddings"):
            projections[i] = np.dot(self.book_embeddings[i], normed_social_axis)
        self._save_projections(projections)

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
        self.pca = PCA(n_components=4)
        self.pca.fit(self.selected_diff_vectors)
        explained_variance = self.pca.explained_variance_ratio_
        pc_labels = [f"PC{i+1}" for i in range(len(explained_variance))]

        plt.figure(figsize=(8, 6))  # 図のサイズを指定（8×6インチ）
        sns.barplot(x=pc_labels, y=explained_variance, color="#636EFA", alpha=0.9)

        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA - Explained Variance Ratio of Principal Components")
        plt.show()
        plt.close()

    def find_gender_axis(
        self,  # 全体で使用するペア数（seed ペア含む）
        n_neighbors: int = 20,  # 各書籍について取得する近傍数（自身を含むので実質10近傍）
        male_book: str = "10億円を捨てた男の仕事術",
        female_book: str = "女子の働き方 男性社会を自由に歩く「自分中心」の仕事術",
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
            if len(selected_pairs) >= self.projection_args.find_axis_pairs:
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
        # 簡単な平均
        # pcaを用いて
        if self.projection_args.how_dim_reduce == "pca":
            pca = PCA(n_components=4)
            pca.fit(self.selected_diff_vectors)
            social_dimension = pca.components_[0]
        else:
            social_dimension = self.selected_diff_vectors.mean(axis=0)
        return social_dimension

    def _compute_diff_vec(self, pair):
        s1, s2 = pair
        return self.book_embeddings[s1] - self.book_embeddings[s2]

    def _read_projection_result(self, result_path: Path):
        return pd.read_csv(result_path)

    def _eval_axis(
        self,
        target_axis,
        male_book: str = "忙しいパパでもできる! 子育てなんとかなるブック (Nanaブックス)",
        female_book: str = "ママも子どもも悪くない!しからずにすむ子育てのヒント",
    ):
        another_axis = self.find_gender_axis(male_book=male_book, female_book=female_book)
        r, p_value = pearsonr(target_axis, another_axis)
        print("\n==社会軸のロバスト性検証==")
        print(f"ピアソン相関係数{r:.3f}")
        print(f"p値：{p_value:.3e}")
        if r < 0.3:
            quit()
