import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR  # noqa
from tqdm import tqdm

from arg import TrainerConfig
from word2vec.preprocessor import BookDataset

from .model import UserBook2Vec


class Trainer(UserBook2Vec):
    def __init__(
        self,
        weight_path: Path,
        grid_search: bool = False,
        dataset: "BookDataset" = None,
        embedding_dim: int = 100,
        num_negatives: int = 5,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 0.005,
        scheduler_factor: float = 0.2,
        early_stop_threshold: float = 0.001,
    ):
        UserBook2Vec.__init__(
            self,
            dataset=dataset,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_negatives=num_negatives,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        self.trainer_config = TrainerConfig()
        self.dataset = dataset
        self.analogy_path = weight_path.parent / "plt" / "movie_analogy.csv"
        self.weight_path = weight_path
        self.weight_path.parent.mkdir(exist_ok=True, parents=True)
        self.grid_search = grid_search

        self.early_stop_threshold = early_stop_threshold
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def train(self):
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=self.scheduler_factor, patience=2
        )
        pairs = np.array(self.pairs)  # shape: (num_pairs, 2)
        num_batches = int(np.ceil(len(pairs) / self.batch_size))
        prev_loss = None

        for epoch in tqdm(range(self.epochs), desc="Epoch Processing", leave=False):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for i in tqdm(range(num_batches), desc="Batch Training", leave=False):
                batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]  # noqa

                user_batch = torch.LongTensor(batch_pairs[:, 0]).to(self.device)
                pos_book_batch = torch.LongTensor(batch_pairs[:, 1]).to(self.device)
                # 各書籍の出現頻度（3/4乗補正）
                book_freq = np.array(
                    [self.book_counts[self.id2book[i]] for i in range(self.vocab_size)]
                )
                book_freq = book_freq**0.75
                book_freq = book_freq / np.sum(book_freq)

                neg_book_batch = torch.LongTensor(
                    self.get_negative_samples(
                        vocab_size=self.vocab_size,
                        book_freq=book_freq,
                        batch_size=batch_pairs.shape[0],
                    )
                ).to(self.device)

                optimizer.zero_grad()
                loss = self(user_batch, pos_book_batch, neg_book_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {avg_loss:.4f}")
            if prev_loss is not None and abs(prev_loss - avg_loss) < self.early_stop_threshold:
                break
            prev_loss = avg_loss

        self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        self.user_embeddings = self.user_embed.weight.data.cpu().numpy()

        id2book = self.dataset.id2book
        vec_df = pd.DataFrame(self.book_embeddings)
        vec_df["title"] = vec_df.index.map(id2book)
        self.vec_df = vec_df.set_index("title")

        if not self.grid_search:
            print("\n=== 学習済み書籍埋め込み (shape) ===", self.book_embeddings.shape)
            self._save_wight_vec()
        else:
            return self.eval_analogy(
                analogy_path=self.dataset.analogy_path, top_range=self.trainer_config.t_range
            )

    def _save_wight_vec(self):
        torch.save(self.state_dict(), self.weight_path)
        print("埋め込みベクトルの書き込みが完了しました")

    def _read_weight_vec(self, device: str, weight_path: Path):
        self.load_state_dict(
            torch.load(weight_path, map_location=torch.device(device), weights_only=True)
        )
        # ベクトルを DataFrame に格納（weight 再読込時）
        id2book = self.dataset.id2book
        self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        self.user_embeddings = self.user_embed.weight.data.cpu().numpy()
        vec_df = pd.DataFrame(self.book_embeddings)
        vec_df["title"] = vec_df.index.map(id2book)
        self.vec_df = vec_df.set_index("title")

    def eval_analogy(
        self,
        analogy_path: Path,
        top_range: int = 100,
    ):
        """
        アナロジータスクを評価し、正解率を返す関数。

        Parameters:
            analogy_path (Path): アナロジータスクのCSVファイルパス
            book2id (dict): 書籍タイトルからIDへの辞書
            id2book (dict): IDから書籍タイトルへの辞書
            book_embeddings (np.ndarray): 書籍ベクトルの埋め込み行列
            top_range (int): 上位何件以内に正解が入っていれば正解とみなすか

        Returns:
            float: 正解率
        """
        analogy_df = pd.read_csv(analogy_path)
        total = 0
        correct = 0

        for _, row in analogy_df.iterrows():
            A, B, C, D = row["A"], row["B"], row["C"], row["D (Answer)"]

            # すべてのタイトルが辞書に含まれていることを確認
            if not all(title in self.dataset.book2id for title in [A, B, C, D]):
                continue

            # アナロジーベクトル計算： vec_B - vec_A + vec_C
            vec_a = self.book_embeddings[self.dataset.book2id[A]]
            vec_b = self.book_embeddings[self.dataset.book2id[B]]
            vec_c = self.book_embeddings[self.dataset.book2id[C]]
            target_vec = vec_b - vec_a + vec_c

            # 類似度の計算（内積で近い順にソート）
            sims = cosine_similarity(target_vec.reshape(1, -1), self.book_embeddings)[0]
            sorted_indices = np.argsort(sims)[::-1]

            # D の index を取得し、上位 top_range に含まれているかを確認
            d_idx = self.dataset.book2id[D]
            top_indices = [
                i
                for i in sorted_indices
                if i != self.dataset.book2id[A]
                and i != self.dataset.book2id[B]
                and i != self.dataset.book2id[C]
            ][:top_range]

            if d_idx in top_indices:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy
