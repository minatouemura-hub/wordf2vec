import json
import random  # noqa
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 文脈-user_id , 単語 - 本のタイトル(著者)
# item2vecに似ているらしい


# データの読み込み
class BookDataset(object):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def _load_dataset(self):
        with open(
            self.folder_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.data_gen = json.load(f)
        if self.data_gen is None:
            raise ValueError(f"{self.folder_path}にデータが入っていません")

    def gender_counter(self):
        male_read_counts = Counter()
        female_read_counts = Counter()
        for uid, books in self.data_gen.items():
            for book in books:
                # 書籍識別子として Title を利用（必要に応じて Author も連結可能）
                book_identifier = f"{book['Title']}"
                # gender 列はユーザーの性別を示していると仮定
                if book["Gender"] == "M":
                    male_read_counts[book_identifier] += 1
                elif book["Gender"] == "F":
                    female_read_counts[book_identifier] += 1
        # --- 条件に基づく書籍の抽出 ---
        # 男性に10回以上読まれ、女性には2回以下の場合
        self.male_fav_books = [
            book
            for book in male_read_counts
            if male_read_counts[book] >= 20 and female_read_counts.get(book, 0) <= 1
        ]

        # 女性に10回以上読まれ、男性には2回以下の場合
        self.female_fav_books = [
            book
            for book in female_read_counts
            if female_read_counts[book] >= 10 and male_read_counts.get(book, 0) <= 1
        ]

        # 結果の出力
        print("\n=== 男性に好まれている書籍（男性: ≥50, 女性: ≤0） ===")
        for book in self.male_fav_books:
            print(f"{book}: 男性{male_read_counts[book]}回, 女性{female_read_counts.get(book,0)}回")

        print("\n=== 女性に好まれている書籍（女性: ≥50, 男性: ≤0） ===")
        for book in self.female_fav_books:
            print(f"{book}: 女性{female_read_counts[book]}回, 男性{male_read_counts.get(book,0)}回")

    def mapping(self):
        ##############################################
        # 2. ユーザーID と 書籍（Title＋Author）のマッピング作成
        ##############################################
        user_id = list(self.data_gen.keys())
        self.user2id = {uid: i for i, uid in enumerate(user_id)}

        book_list = []
        for uid, books in self.data_gen.items():
            for book in books:
                title = book["Title"]
                book_identifier = f"{title}"
                book_list.append(book_identifier)

        self.book_counts = Counter(book_list)
        self.book2id = {}
        self.id2book = {}

        for i, book in enumerate(self.book_counts.keys()):
            self.book2id[book] = i
            self.id2book[i] = book

        ##############################################
        # 3. 学習データの作成： (user, book) ペア
        ##############################################

        self.pairs = []
        for uid, books in tqdm(self.data_gen.items(), desc="Processing Pairing"):
            user_idx = self.user2id[uid]
            for book in books:
                book_identifier = book["Title"]
                book_idx = self.book2id[book_identifier]
                self.pairs.append((user_idx, book_idx))


##############################################
# 5. PyTorch によるユーザー・書籍埋め込みモデルの定義
##############################################


# 埋め込み表現
class UserBook2Vec(nn.Module, BookDataset):
    def __init__(
        self,
        folder_path: str,
        embedding_dim: int = 50,
        num_negatives: int = 5,
        batch_size: int = 4,
        epochs: int = 10,
        learning_rate: int = 0.005,
    ):
        # まず nn.Module の初期化を行う
        nn.Module.__init__(self)
        # BookDataset の初期化も行う（UserBook2Vec が BookDataset を継承しているので）
        BookDataset.__init__(self, folder_path)

        # データの読み込み
        self._load_dataset()
        self.gender_counter()
        self.mapping()

        self.num_users = len(self.user2id)
        self.vocab_size = len(self.book2id)

        # ユーザーの埋め込み（文脈側）
        self.user_embed = nn.Embedding(self.num_users, embedding_dim)
        # 書籍の埋め込み（ターゲット側）
        self.book_embed = nn.Embedding(self.vocab_size, embedding_dim)

        # 初期設定
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, user_ids, pos_book_ids, neg_book_ids):
        # user_ids: (batch_size,)
        # pos_book_ids: (batch_size,)
        # neg_book_ids: (batch_size, num_negatives)
        user_vec = self.user_embed(user_ids)  # (batch_size, embedding_dim)
        pos_book_vec = self.book_embed(pos_book_ids)  # (batch_size, embedding_dim)
        neg_book_vec = self.book_embed(neg_book_ids)  # (batch_size, num_negatives, embedding_dim)

        # 正例スコア（内積）
        pos_score = torch.sum(user_vec * pos_book_vec, dim=1)  # (batch_size,)
        pos_score = torch.log(torch.sigmoid(pos_score) + 1e-10)

        # 負例スコア
        neg_score = torch.bmm(neg_book_vec, user_vec.unsqueeze(2)).squeeze(
            -1
        )  # (batch_size, num_negatives)
        neg_score = torch.log(torch.sigmoid(-neg_score) + 1e-10)
        neg_score = torch.sum(neg_score, dim=1)  # (batch_size,)

        loss = -(pos_score + neg_score)
        return loss.mean()

    def get_negative_samples(self, vocab_size, book_freq, batch_size):
        negatives = np.random.choice(vocab_size, size=(batch_size, self.num_negatives), p=book_freq)
        return negatives


# 訓練
class Trainer(UserBook2Vec):
    def __init__(
        self,
        folder_path: Path,
        weight_path: Path,
        embedding_dim=50,
        num_negatives=5,
        batch_size=32,
        epochs=5,
        learning_rate=0.005,
    ):
        UserBook2Vec.__init__(self, folder_path=folder_path)

        self.weight_path = weight_path

        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        pairs = np.array(self.pairs)  # shape: (num_pairs, 2)
        num_batches = int(np.ceil(len(pairs) / self.batch_size))

        print("\n=== 訓練開始 ===")
        for epoch in tqdm(range(self.epochs), desc="Epoch Processing", leave=False):
            np.random.shuffle(pairs)
            total_loss = 0.0
            for i in tqdm(range(num_batches), desc="Batch Training", leave=False):
                batch_pairs = pairs[i * self.batch_size : (i + 1) * self.batch_size]  # noqa

                user_batch = torch.LongTensor(batch_pairs[:, 0]).to(device)
                pos_book_batch = torch.LongTensor(batch_pairs[:, 1]).to(device)
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
                ).to(device)

                optimizer.zero_grad()
                loss = self(user_batch, pos_book_batch, neg_book_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss/num_batches:.4f}")

        ##############################################
        # 7. 学習結果の確認：書籍埋め込みと識別子対応
        ##############################################
        # 学習後、書籍側の埋め込みを取得
        self.book_embeddings = self.book_embed.weight.data.cpu().numpy()

        print("\n=== 学習済み書籍埋め込み (shape) ===", self.book_embeddings.shape)
        self._save_wight_vec()

    def find_gender_axis(self, top_k: int = 5):
        male_fav_bookid = self.book2id[self.male_fav_books[0]]
        female_fav_bookdid = self.book2id[self.female_fav_books[0]]

        # 1.
        male_end = torch.tensor(self.book_embeddings[male_fav_bookid], dtype=torch.float32)  # noqa
        female_end = torch.tensor(  # noqa
            self.book_embeddings[female_fav_bookdid], dtype=torch.float32
        )
        first_pair = (male_fav_bookid, female_fav_bookdid)
        seed_diff = self._compute_diff_vec(first_pair)
        seed_diff = seed_diff.reshape(1, -1)
        # 2.
        books = list(range(len(self.book_embeddings)))
        self.candidates_pair = []
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                pair = (books[i], books[j])
                if set(pair) == set(first_pair):
                    continue
                self.candidates_pair.append(pair)

        pair_sims = []
        for pair in self.candidates_pair:
            diff_vec = self._compute_diff_vec()
            diff_vec = diff_vec.reshape(1, -1)
            sim = cosine_similarity(seed_diff, diff_vec)[0, 0]
            pair_sims.append((pair, sim))

        pair_sims.sort(key=lambda x: x[1], reverse=True)

        # 上位 top_k 件を選択
        selected_pairs = [first_pair]  # シードペアを必ず含む
        for pair, sim in pair_sims[:top_k]:
            selected_pairs.append(pair)

        print("Selected pairs:")
        for pair, sim in pair_sims[:top_k]:
            print(f"{pair}: sim={sim:.4f}")

        # 選択された各ペアの差分ベクトルを計算し平均する
        diff_vectors = []
        for pair in selected_pairs:
            diff_vec = self._compute_diff_vec(self.book_embeddings, pair)
            diff_vectors.append(diff_vec)
        diff_vectors = np.array(diff_vectors)
        social_dimension = diff_vectors.mean(axis=0)
        # Genderの軸は以下の形で保管されています
        print(social_dimension.shape)
        return social_dimension

    def _compute_diff_vec(self, pair):
        s1, s2 = pair
        return self.book_embeddings[s1] - self.book_embeddings[s2]

    def _save_wight_vec(self):
        torch.save(self.state_dict(), self.weight_path)
        print("埋め込みベクトルの書き込みが完了しました")

    def read_weight_vec(self, device: str):
        self.load_state_dict(torch.load(self.weight_path, map_location=torch.device(device)))
        self.eval()
