import json
import random  # noqa
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
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
    def __init__(self, folder_path: str, down_sample: bool = True, sample: float = 1e-4):
        self.folder_path = folder_path
        self.down_sample = down_sample
        self.sample = sample

    def execute(self):
        self._load_dataset()
        self._norm_titles()
        self.gender_counter()
        self.mapping()

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
        number_pattern = r"(?:\d+|[零一二三四五六七八九十百千万億兆]+)"
        patterns = [
            (r"\s*全\s*\d+\s*巻(セット|完結)?(?:\s*\[.*?\])?", ""),  # より汎用的に
            # 新規パターン: タイトルが【...】で始まり、「~」がある場合、最初の「~」以降を除去する
            (r"^(?:【.*?】)\s*(.+?~.+?~).*$", r"\1"),
            # 新規パターン: タイトル末尾に空白＋数字＋括弧内の不要情報がある場合、先頭部分のみ抽出
            (r"^(.*?)\s+\d+\s*\(.*\)$", r"\1"),
            # 以下、既存のパターン群
            # 末尾にある、括弧（または類似記号）で囲まれた「第」(任意)＋数字または[上下]+＋(巻)があれば削除
            (r"[\(〈<]?(?:第)?\s*(?:" + number_pattern + r"|[上下]+)(?:巻)?[\)〉>].*$", ""),
            (r"^(.*?)\s*[\(〈<]?\s*[上下]\s*[\)〉>]?(?:\s+.*)?$", r"\1"),
            (r"^(.*?)\s*(?:[\(]?(?:第)?\d+巻[\)]?).*$", r"\1"),
            # パターン0: 上巻/下巻の場合：タイトルと、上巻または下巻（括弧や山括弧の有無は問わない）およびその後の余分な情報を除去
            # パターン① 数字の場合：タイトルと、数字部分（括弧・山括弧の有無は問わない）、およびその後の余分な情報を除去
            (r"^(.*?)\s*[\(〈<]?\s*\d+\s*[\)〉>]?(?:\s+.*)?$", r"\1"),
        ]
        for user, books in self.data_gen.items():
            for book in books:
                title = book.get("Title", "")
                title = self._remove_sub_title(self._convert_brackets(title))
                book["Title"] = title
                for pattern, replace in patterns:
                    norm_title = self._remove_sub_title(
                        re.sub(pattern, replace, title, flags=re.IGNORECASE)
                    )
                    if norm_title != title:
                        book["Title"] = self._remove_bunko_prefix(norm_title.strip())
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

    def _remove_bunko_prefix(self, title: str) -> str:
        """
        タイトルの先頭にある「文庫」とその後の空白を除去する。
        """
        return re.sub(r"^文庫\s*", "", title)

    def _remove_sub_title(self, title: str) -> str:
        return re.sub(r"^(.*?~.*?~).*$", r"\1", title)

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
        self.pairs = []
        if self.down_sample:
            total_count = sum(self.book_counts.values())
            book_freq = {b: c / total_count for b, c in self.book_counts.items()}
            self.book_keep_prob = {
                b: 1 - np.sqrt(self.sample / f) if f > self.sample else 1.0
                for b, f in book_freq.items()
            }
        for uid, books in tqdm(self.data_gen.items(), desc="Processing Pairing"):
            user_idx = self.user2id[uid]
            for book in books:
                book_identifier = book["Title"]
                if self.down_sample and np.random.rand() < self.book_keep_prob.get(
                    book_identifier, 1.0
                ):
                    continue
                book_idx = self.book2id[book_identifier]
                self.pairs.append((user_idx, book_idx))


# 埋め込み表現
class UserBook2Vec(nn.Module):
    def __init__(
        self,
        folder_path: str,
        dataset: "BookDataset" = None,
        embedding_dim: int = 50,
        num_negatives: int = 5,
        batch_size: int = 4,
        epochs: int = 10,
        learning_rate: int = 0.005,
    ):
        # まず nn.Module の初期化を行う
        nn.Module.__init__(self)
        # BookDataset の初期化も行う（UserBook2Vec が BookDataset を継承しているので）
        if dataset is not None:
            self.__dict__.update(dataset.__dict__)
        else:
            BookDataset.__init__(self, folder_path)
            self.execute()

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
        dataset: "BookDataset" = None,
        embedding_dim=50,
        num_negatives=5,
        batch_size=32,
        epochs=5,
        learning_rate=0.005,
    ):
        UserBook2Vec.__init__(
            self,
            folder_path=folder_path,
            dataset=dataset,
            batch_size=batch_size,
            num_negatives=num_negatives,
            epochs=epochs,
            learning_rate=learning_rate,
        )

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
        acc = self._eval_analogy(analogy_path=self.folder_path.parent / "analogy_task.csv")
        return acc

    def _save_wight_vec(self):
        torch.save(self.state_dict(), self.weight_path)
        print("埋め込みベクトルの書き込みが完了しました")

    def _read_weight_vec(self, device: str):
        self.load_state_dict(
            torch.load(self.weight_path, map_location=torch.device(device), weights_only=True)
        )
        self.eval()

    def _eval_analogy(self, analogy_path: Path, thereshold: float = 0.6):
        analogy_df = pd.read_csv(analogy_path)

        total = len(analogy_df)
        correct = 0

        for _, row in analogy_df.iterrows():
            analogy_list = [row["A"], row["B"], row["C"], row["D"]]
            for analogy_title in analogy_list:
                if analogy_title not in self.book2id.keys():
                    raise ValueError(f"{analogy_title} not in book_dict")

            vec_a = self.book_embeddings[self.book2id[analogy_list[0]]]
            vec_b = self.book_embeddings[self.book2id[analogy_list[1]]]
            vec_c = self.book_embeddings[self.book2id[analogy_list[2]]]
            vec_d = self.book_embeddings[self.book2id[analogy_list[3]]]

            predic_vec = vec_b - vec_a + vec_c

            sim = cosine_similarity(predic_vec.reshape(1, -1), vec_d.reshape(1, -1))
            if sim > thereshold:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        print(sim)
        print(f"Accuracy:{acc} based on threshold:{thereshold}")
        return acc
