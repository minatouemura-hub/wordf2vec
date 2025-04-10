from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR  # noqa
from tqdm import tqdm

from word2vec.preprocessor import BookDataset

from .model import UserBook2Vec


class Trainer(UserBook2Vec):
    def __init__(
        self,
        folder_path: Path,
        weight_path: Path,
        dataset: "BookDataset" = None,
        embedding_dim: int = 100,
        num_negatives: int = 5,
        batch_size: int = 32,
        epochs: int = 5,
        learning_rate: float = 0.005,
        scheduler_factor: float = 0.2,
        early_stop_threshold: float = 0.001,
        top_range: int = 5,
        eval_task: str = "sim_task",
    ):
        UserBook2Vec.__init__(
            self,
            folder_path=folder_path,
            dataset=dataset,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_negatives=num_negatives,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        self.dataset = dataset
        self.eval_task = eval_task
        self.weight_path = weight_path
        self.early_stop_threshold = early_stop_threshold
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.top_range = top_range

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
            scheduler.step(total_loss)
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss/num_batches:.4f}")
            if prev_loss is not None and abs(prev_loss - total_loss) < self.early_stop_threshold:
                break
            prev_loss = total_loss

        self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        id2book = self.dataset.id2book
        vec_df = pd.DataFrame(self.book_embeddings)
        vec_df["title"] = vec_df.index.map(id2book)
        self.vec_df = vec_df.set_index("title")

        print("\n=== 学習済み書籍埋め込み (shape) ===", self.book_embeddings.shape)
        self._save_wight_vec()

        # if self.eval_task == "analogy_task":
        #     acc = self._eval_analogy(
        #         analogy_path=self.folder_path.parent / f"{self.eval_task}.csv",
        #     )
        # else:
        #     acc = self._eval_sim(sim_path=self.folder_path.parent / f"{self.eval_task}.csv")
        # return acc

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
        vec_df = pd.DataFrame(self.book_embeddings)
        vec_df["title"] = vec_df.index.map(id2book)
        self.vec_df = vec_df.set_index("title")

    def _eval_analogy(self, analogy_path: Path):
        analogy_df = pd.read_csv(analogy_path)
        total = len(analogy_df)
        correct = 0
        for _, row in analogy_df.iterrows():
            analogy_list = [row["A"], row["B"]]
            for analogy_title in analogy_list:
                if analogy_title not in self.book2id.keys():
                    print(f"{analogy_title} not in book_dict")
                    continue

            vec_a = self.book_embeddings[self.book2id[analogy_list[0]]]
            vec_b = self.book_embeddings[self.book2id[analogy_list[1]]]
            # vec_c = self.book_embeddings[self.book2id[analogy_list[2]]]
            # vec_d = self.book_embeddings[self.book2id[analogy_list[3]]]

            # predic_vec = vec_b - vec_a + vec_c

            sims = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0]
            print(sims)
            if sims >= 0.6:
                correct += 1

        #     a_idx = self.book2id[analogy_list[0]]
        #     top_indices = [i for i in sims.argsort()[::-1] if i != a_idx][: self.top_range]

        #     target_idx = self.book2id[analogy_list[3]]
        #     top_titles = [self.id2book[i] for i in top_indices]
        #     if target_idx in top_indices:
        #         correct += 1
        #     print(f"Top-{self.top_range} Predictions: {top_titles}")
        acc = correct / total if total > 0 else 0.0
        print(f"Accuracy:{acc} based on top_range{self.top_range}")
        return acc

    def _eval_sim(self, sim_path: Path, threshold: float = 0.6):
        sim_df = pd.read_csv(sim_path)
        total = len(sim_df)
        correct = 0
        for _, row in sim_df.iterrows():
            if row["A"] not in self.book2id.keys() or row["B"] not in self.book2id.keys():
                total -= 1

                continue
            vec_a = self.book_embeddings[self.book2id[row["A"]]]
            vec_b = self.book_embeddings[self.book2id[row["B"]]]
            similality = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))
            if similality > threshold:
                correct += 1
            print(similality)
        acc = correct / total if total > 0 else 0.0
        print(f"Accuracy:{acc} based on {threshold}")
        return acc
