import json
import os
import random  # noqa
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class BookDataset(object):
    def __init__(
        self,
        folder_path: Path,
        word_col: str = "Title",
        down_sample: bool = True,
        sample: float = 1e-4,
        min_user_cnt: int = 3,
    ):
        self.folder_path = folder_path
        self.word_col = word_col
        self.down_sample = down_sample
        self.sample = sample
        self.min_user_cnt = min_user_cnt

    def preprocess(self):
        self._load_dataset()
        self._norm_titles()
        self.gender_counter()
        self.mapping()
        self.make_analogy(base_dir=self.folder_path.parent.parent, n_samples=100)

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
            (r"\s*\(.*?\)$", ""),
        ]
        for user, books in self.data_gen.items():
            for book in books:
                title = book.get("Title", "")
                title = re.sub(r"\s*\(.*?\)$", "", title)
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

        normalized_unique_count = len(set(normalized_titles))

        print(
            f"Normalization後: 総書籍数={len(normalized_titles)}, ユニークタイトル数={normalized_unique_count}"
        )

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

    def gender_counter(self, target_col: str = "Title"):
        male_user_per_book = Counter()
        female_user_per_book = Counter()

        # 各本に対して、読んだ男性ユーザー・女性ユーザーのセットを構築
        book_male_users = {}
        book_female_users = {}

        for uid, books in self.data_gen.items():
            gender = None
            for book in books:
                gender = book.get("Gender")
                book_identifier = f"{book[target_col]}"
                if gender == "M":
                    book_male_users.setdefault(book_identifier, set()).add(uid)
                elif gender == "F":
                    book_female_users.setdefault(book_identifier, set()).add(uid)

        # ユニークユーザー数をカウント
        for book, users in book_male_users.items():
            male_user_per_book[book] = len(users)
        for book, users in book_female_users.items():
            female_user_per_book[book] = len(users)

        self.male_fav_books = [
            book
            for book in male_user_per_book
            if 100 >= male_user_per_book[book] >= 10 and female_user_per_book.get(book, 0) <= 1
        ]

        self.female_fav_books = [
            book
            for book in female_user_per_book
            if 100 >= female_user_per_book[book] >= 10 and male_user_per_book.get(book, 0) <= 1
        ]

        print(
            f"\n=== 男性に好まれている{target_col}（男性ユーザー数: ≥10, 女性ユーザー数: ≤1） ==="
        )
        for book in self.male_fav_books:
            print(
                f"{book}: 男性{male_user_per_book[book]}人, 女性{female_user_per_book.get(book,0)}人"
            )

        print(
            f"\n=== 女性に好まれている{target_col}（女性ユーザー数: ≥10, 男性ユーザー数: ≤1） ==="
        )
        for book in self.female_fav_books:
            print(
                f"{book}: 女性{female_user_per_book[book]}人, 男性{male_user_per_book.get(book,0)}人"
            )

    def mapping(self, target_col: str = "Title"):

        user_id = list(self.data_gen.keys())
        self.user2id = {uid: i for i, uid in enumerate(user_id)}

        # ユーザー間で少なくともmin_user_cnt回以上出現しているもの
        user_cnt_per_books = Counter()
        for uid, books in self.data_gen.items():
            uniq_books = set(book[target_col] for book in books)
            for book in uniq_books:
                user_cnt_per_books[book] += 1
        filtered_books = {b for b, c in user_cnt_per_books.items() if c >= self.min_user_cnt}

        # 保存先のディレクトリを作成（必要なら）
        output_path = self.folder_path.parent.parent / "result"
        output_path.mkdir(parents=True, exist_ok=True)

        # 正規化後タイトルをJSONで保存（リスト形式）
        with open(output_path / "normed_title.json", "w", encoding="utf-8") as f:
            json.dump(list(filtered_books), f, ensure_ascii=False, indent=2)

        book_list = []

        for uid, books in self.data_gen.items():
            for book in books:
                title = book[target_col]
                book_identifier = f"{title}"
                if title in filtered_books:
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
                book_identifier = book[target_col]
                if self.down_sample and np.random.rand() < self.book_keep_prob.get(
                    book_identifier, 1.0
                ):
                    continue
                book_idx = self.book2id[book_identifier]
                self.pairs.append((user_idx, book_idx))

        print(f"フィルタ後の学習対象書籍数（ユニーク）: {len(self.book2id)}")

    def make_analogy(self, base_dir: Path, n_samples: int = 100):
        """
        作者情報に基づいたアナロジータスクを作成し、CSVに保存する。

        Parameters:
            base_dir (Path): 出力先フォルダ
            n_samples (int): 作成するアナロジータスク数（デフォルト100件）
        """
        author_to_titles = defaultdict(list)

        for _, books in self.data_gen.items():
            for book in books:
                author = book.get("Author")
                title = book.get("Title")
                if author and title:
                    author_to_titles[author].append(title)

        # 有効なアナロジータスクを作れる作者を抽出
        valid_authors = [a for a, titles in author_to_titles.items() if len(set(titles)) >= 4]

        analogy_tasks = []
        for _ in range(n_samples * 2):  # 多めに生成してフィルタ
            if len(valid_authors) < 2:
                break
            author1, author2 = random.sample(valid_authors, 2)
            titles1 = list(set(author_to_titles[author1]))
            titles2 = list(set(author_to_titles[author2]))

            if len(titles1) < 2 or len(titles2) < 2:
                continue

            A, B = random.sample(titles1, 2)
            C, D = random.sample(titles2, 2)

            analogy_tasks.append(
                {
                    "Author1": author1,
                    "Author2": author2,
                    "A": A,
                    "B": B,
                    "C": C,
                    "D (Answer)": D,
                    "Analogy": f"({A} : {B}) = ({C} : ?)",
                }
            )

            if len(analogy_tasks) >= n_samples:
                break

        analogy_dir = base_dir / "analogy"
        analogy_dir.mkdir(parents=True, exist_ok=True)
        df_analogy = pd.DataFrame(analogy_tasks)
        self.analogy_path = analogy_dir / "book_analogy.csv"
        df_analogy.to_csv(self.analogy_path, index=False)
        return df_analogy


class Movie1MDataset(object):
    def __init__(
        self,
        movie_dir: Path,
        down_sample: bool = True,
        sample: float = 1e-4,
        min_user_cnt: int = 3,
    ):
        self.movie_dir = movie_dir
        self.down_sample = down_sample
        self.sample = sample
        self.min_user_cnt = min_user_cnt

    def preprocess(self):
        self._load_dataset()
        self.mapping()
        self.make_analogy(base_dir=self.movie_dir.parent, n_samples=100)

    def _load_dataset(self):
        if not os.path.isdir(self.movie_dir):
            raise FileExistsError(
                "Please Download Dataset from https://grouplens.org/datasets/movielens/1m/"
            )
        # データの読み込み
        ratings = pd.read_csv(
            self.movie_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )

        users = pd.read_csv(
            self.movie_dir / "users.dat",
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip"],
        )

        movies = pd.read_csv(
            self.movie_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )

        self.data_gen = ratings.merge(users, on="userId").merge(movies, on="movieId")

    def mapping(self, target_col: str = "title"):
        user_id = list(self.data_gen["userId"].unique())
        self.user2id = {uid: i for i, uid in enumerate(user_id)}

        # 各映画が何人のユーザーに見られたかをカウント（重複なし）
        user_cnt_per_movie = Counter()
        for uid, group in self.data_gen.groupby("userId"):
            uniq_titles = set(group[target_col])
            for title in uniq_titles:
                user_cnt_per_movie[title] += 1

        # 最低視聴ユーザー数を満たす映画だけを残す
        filtered_titles = {
            title for title, cnt in user_cnt_per_movie.items() if cnt >= self.min_user_cnt
        }

        # 保存先
        output_path = Path(self.movie_dir).parent / "result"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "normed_title.json", "w", encoding="utf-8") as f:
            json.dump(list(filtered_titles), f, ensure_ascii=False, indent=2)

        # タイトルごとの出現数カウント（学習で使う）
        filtered_df = self.data_gen[self.data_gen[target_col].isin(filtered_titles)]
        movie_list = filtered_df[target_col].tolist()
        self.book_counts = Counter(movie_list)

        # title <-> ID の対応表
        self.book2id = {title: i for i, title in enumerate(self.book_counts)}
        self.id2book = {i: title for title, i in self.book2id.items()}

        # サンプリング確率の計算（出現頻度に基づく）
        self.pairs = []
        if self.down_sample:
            total = sum(self.book_counts.values())
            book_freq = {b: c / total for b, c in self.book_counts.items()}
            self.book_keep_prob = {
                b: 1 - np.sqrt(self.sample / f) if f > self.sample else 1.0
                for b, f in book_freq.items()
            }

        # ユーザーと映画のペア作成
        for uid, group in tqdm(self.data_gen.groupby("userId"), desc="Pairing by Title"):
            user_idx = self.user2id[uid]
            for title in group[target_col]:
                if title not in self.book2id:
                    continue
                if self.down_sample and np.random.rand() < self.book_keep_prob.get(title, 1.0):
                    continue
                book_idx = self.book2id[title]
                self.pairs.append((user_idx, book_idx))

        print(f"フィルタ後の映画数（ユニークタイトル）: {len(self.book2id)}")

    def make_analogy(self, base_dir: Path, n_samples: int = 100):
        """
        ジャンルと年代の組み合わせに基づいたアナロジータスクを作成し、CSVに保存する。

        Parameters:
            base_dir (Path): CSV出力先のベースパス（.csvファイルを含むフォルダ）
            n_samples (int): アナロジータスク数（デフォルト100件）
        """

        def extract_decade(title):
            match = re.search(r"\((\d{4})\)$", title)
            if match:
                year = int(match.group(1))
                return (year // 10) * 10
            return None

        contrasting_genre_pairs = [
            (frozenset(["Comedy"]), frozenset(["Horror"])),
            (frozenset(["Children's"]), frozenset(["Horror"])),
            (frozenset(["Musical"]), frozenset(["Action"])),
            (frozenset(["Crime"]), frozenset(["Animation"])),
            (frozenset(["Romance"]), frozenset(["Action"])),
            (frozenset(["Documentary"]), frozenset(["Fantasy"])),
            (frozenset(["Drama"]), frozenset(["Sci-Fi"])),
        ]

        genre_decade_to_titles = defaultdict(list)
        for _, row in self.data_gen.iterrows():
            decade = extract_decade(row["title"])
            if decade is None:
                continue
            genre_set = frozenset(row["genres"].split("|"))
            genre_decade_to_titles[(genre_set, decade)].append(row["title"])

        analogy_tasks = []

        for genre1, genre2 in contrasting_genre_pairs:
            for decade in range(1960, 2020, 10):
                titles1 = genre_decade_to_titles.get((genre1, decade), [])
                titles2 = genre_decade_to_titles.get((genre2, decade), [])
                if len(titles1) < 3 or len(titles2) < 3:
                    continue
                for _ in range(3):
                    A, B = random.sample(titles1, 2)
                    C, D = random.sample(titles2, 2)
                    analogy_tasks.append(
                        {
                            "Genre1": "|".join(sorted(genre1)),
                            "Genre2": "|".join(sorted(genre2)),
                            "Decade": f"{decade}s",
                            "A": A,
                            "B": B,
                            "C": C,
                            "D (Answer)": D,
                            "Analogy": f"({A} : {B}) = ({C} : ?)",
                        }
                    )
                    if len(analogy_tasks) >= n_samples:
                        break
                if len(analogy_tasks) >= n_samples:
                    break
            if len(analogy_tasks) >= n_samples:
                break

        analogy_dir = base_dir / "analogy"
        analogy_dir.mkdir(parents=True, exist_ok=True)
        df_analogy = pd.DataFrame(analogy_tasks[:n_samples])
        self.analogy_path = analogy_dir / "movie_analogy.csv"
        df_analogy.to_csv(self.analogy_path, index=False)
        return df_analogy


class Movie10MDataset(object):
    """
    Movielens 10M データセット処理クラス
    ratings.dat と movies.dat を読み込み、学習用ペアとアナロジータスクを生成します。
    """
    def __init__(
        self,
        movie_dir: Path,
        down_sample: bool = True,
        sample: float = 1e-4,
        min_user_cnt: int = 3,
    ):
        self.movie_dir = movie_dir
        self.down_sample = down_sample
        self.sample = sample
        self.min_user_cnt = min_user_cnt

    def preprocess(self):
        self._load_dataset()
        self.mapping()
        self.make_analogy(base_dir=self.movie_dir.parent, n_samples=100)

    def _load_dataset(self):
        if not os.path.isdir(self.movie_dir):
            raise FileExistsError(
                "10M データセットが見つかりません。Movielens 10M データをダウンロードし、movie_dir に配置してください。"
            )
        # ratings.dat の読み込み
        ratings = pd.read_csv(
            self.movie_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )
        # movies.dat の読み込み
        movies = pd.read_csv(
            self.movie_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        # マージ
        self.data_gen = ratings.merge(movies, on="movieId")

    def mapping(self, target_col: str = "title"):
        # 各映画のユニーク視聴ユーザー数をカウント
        user_cnt_per_movie = Counter()
        for uid, group in self.data_gen.groupby("userId"):
            uniq_titles = set(group[target_col])
            for title in uniq_titles:
                user_cnt_per_movie[title] += 1

        # 最低視聴ユーザー数を満たす映画を抽出
        filtered_titles = {
            title for title, cnt in user_cnt_per_movie.items() if cnt >= self.min_user_cnt
        }

        # 出力先ディレクトリ
        output_path = Path(self.movie_dir).parent / "result"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "normed_title_10m.json", "w", encoding="utf-8") as f:
            json.dump(list(filtered_titles), f, ensure_ascii=False, indent=2)

        # 学習用ペア生成
        self.book2id = {title: idx for idx, title in enumerate(filtered_titles)}
        self.id2book = {idx: title for title, idx in self.book2id.items()}

        # ダウンサンプリング確率
        if self.down_sample:
            counts = sum(user_cnt_per_movie[t] for t in filtered_titles)
            freq = {t: user_cnt_per_movie[t] / counts for t in filtered_titles}
            self.book_keep_prob = {
                t: 1 - np.sqrt(self.sample / f) if f > self.sample else 1.0 for t, f in freq.items()
            }

        self.pairs = []
        for uid, group in tqdm(self.data_gen.groupby("userId"), desc="Pairing (10M)"):
            user_idx = uid  # raw userId をそのまま使用可能
            for title in group[target_col]:
                if title not in self.book2id:
                    continue
                if self.down_sample and np.random.rand() < self.book_keep_prob.get(title, 1.0):
                    continue
                book_idx = self.book2id[title]
                self.pairs.append((user_idx, book_idx))

        print(f"フィルタ後の映画数（ユニーク）：{len(self.book2id)} 件")

    def make_analogy(self, base_dir: Path, n_samples: int = 100):
        """
        高速化：Pandas の groupby とサンプリングで O(n_samples) に近い計算量でタスク生成
        """
        # データコピーと前処理
        df = self.data_gen.copy()
        # 年代抽出
        df['year'] = df['title'].str.extract(r"\((\d{4})\)$")[0].astype(float)
        df = df.dropna(subset=['year'])
        df['decade'] = (df['year']//10*10).astype(int)
        # ジャンルセット化
        df['genres_set'] = df['genres'].str.split('|').apply(frozenset)
        # グループ化
        grouped = df.groupby(['genres_set','decade'])['title'].unique()
        valid = grouped[grouped.apply(lambda x: len(x)>=4)]
        keys = list(valid.index)
        if not keys:
            return pd.DataFrame()
        tasks = []
        for _ in range(n_samples):
            genres, decade = random.choice(keys)
            titles = valid[(genres,decade)]
            A, B, C, D = random.sample(list(titles), 4)
            tasks.append({
                'Genre': '|'.join(sorted(genres)),
                'Decade': f"{decade}s",
                'A': A, 'B': B, 'C': C, 'D (Answer)': D,
                'Analogy': f"({A} : {B}) = ({C} : ?)"
            })
        df_tasks = pd.DataFrame(tasks)
        analogy_dir = base_dir / "analogy_10m"
        analogy_dir.mkdir(parents=True, exist_ok=True)
        self.analogy_path = analogy_dir / "movie10m_analogy.csv"
        df_tasks.to_csv(self.analogy_path, index=False)
        return df_tasks

