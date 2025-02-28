import json
import random  # noqa
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
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

    def _norm_titles(self):
        # 処理前の総書籍数
        original_titles = []
        for user, books in self.data_gen.items():
            for book in books:
                original_titles.append(book.get("Title", ""))
        original_unique_count = len(set(original_titles))
        print("==タイトルの正規化を始めます==")
        print(
            f"Normalization前: 総書籍数={len(original_titles)}, ユニークタイトル数={original_unique_count}"
        )
        patterns = [
            # パターン0: 上巻/下巻の場合：タイトルと、上巻または下巻（括弧や山括弧の有無は問わない）およびその後の余分な情報を除去
            (r"^(.*?)\s*[\(〈<]?\s*(上巻|下巻)\s*[\)〉>]?(?:\s+.*)?$", r"\1"),
            # パターン① 数字の場合：
            # タイトルと、数字部分（括弧・山括弧の有無は問わない）、およびその後の余分な情報を除去
            (r"^(.*?)\s*[\(〈<]?\s*\d+\s*[\)〉>]?(?:\s+.*)?$", r"\1"),
            # パターン② 上下の場合：
            # タイトルと、上または下（括弧・山括弧の有無は問わない）、およびその後の余分な情報を除去
            (r"^(.*?)\s*[\(〈<]?\s*[上下]\s*[\)〉>]?(?:\s+.*)?$", r"\1"),
            (r"^(.*?)\s*(?:[\(]?(?:第)?\d+巻[\)]?).*$", r"\1"),
        ]
        for user, books in self.data_gen.items():
            for book in books:
                for pattern, replace in patterns:
                    title = book.get("Title", "")
                    title = self._convert_brackets(title)
                    norm_title = re.sub(pattern, replace, title, flags=re.IGNORECASE)
                    if norm_title != title:
                        book["Title"] = norm_title
                        break
        # 処理後のユニークなタイトル数をカウント
        normalized_titles = []
        for user, books in self.data_gen.items():
            for book in books:
                normalized_titles.append(book.get("Title", ""))
        normalized_unique_count = len(set(normalized_titles))
        print(
            f"Normalization後: 総書籍数={len(normalized_titles)}, ユニークタイトル数={normalized_unique_count}"
        )
        print("====")

    def _convert_brackets(self, text: str):
        norm_text = (
            text.replace("（", "(")
            .replace("）", ")")
            .replace("＜", "<")
            .replace("＞", ">")
            .replace("　", " ")
        )
        return unicodedata.normalize("NFKC", norm_text)

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
            if male_read_counts[book] >= 100 and female_read_counts.get(book, 0) <= 1
        ]

        # 女性に10回以上読まれ、男性には2回以下の場合
        self.female_fav_books = [
            book
            for book in female_read_counts
            if female_read_counts[book] >= 100 and male_read_counts.get(book, 0) <= 1
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
        # 2. ユーザーID と 書籍（Title）のマッピング作成
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
        self._norm_titles()
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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def run_train(self):
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        pairs = np.array(self.pairs)  # shape: (num_pairs, 2)
        num_batches = int(np.ceil(len(pairs) / self.batch_size))

        print("\n=== 訓練開始 ===")
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
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {total_loss/num_batches:.4f}")

        ##############################################
        # 7. 学習結果の確認：書籍埋め込みと識別子対応
        ##############################################
        # 学習後、書籍側の埋め込みを取得
        self.book_embeddings = self.book_embed.weight.data.cpu().numpy()

        print("\n=== 学習済み書籍埋め込み (shape) ===", self.book_embeddings.shape)
        self._save_wight_vec()

    def find_gender_axis(
        self,
        k: int = 10,  # 全体で使用するペア数（seed ペア含む）
        n_neighbors: int = 10,  # 各書籍について取得する近傍数（自身を含むので実質10近傍）
        male_book: str = "人間の建設 (新潮文庫)",
        female_book: str = "生理用品の社会史: タブーから一大ビジネスへ",
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
        selected_diff_vectors = np.array(
            [self._compute_diff_vec(pair) for pair, _ in selected_pairs]
        )
        social_dimension = selected_diff_vectors.mean(axis=0)
        return social_dimension

    def _compute_diff_vec(self, pair):
        s1, s2 = pair
        return self.book_embeddings[s1] - self.book_embeddings[s2]

    def _save_wight_vec(self):
        torch.save(self.state_dict(), self.weight_path)
        print("埋め込みベクトルの書き込みが完了しました")

    def _read_weight_vec(self, device: str):
        self.load_state_dict(
            torch.load(self.weight_path, map_location=torch.device(device), weights_only=True)
        )
        self.eval()
