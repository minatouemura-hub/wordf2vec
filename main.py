import argparse  # noqa
import os  # noqa
import sys  # noqa

# 上記設定を行ったあとにnumpyやscipyなどをインポート
import warnings
from pathlib import Path
from typing import Any, Dict

import pandas as pd  # noqa
import torch
from sklearn.cluster import KMeans

from arg import get_args, parse_config
from cluster_analysis.analysis import Balanced_Kmeans  # noqa
from cluster_analysis.analysis import (
    evaluate_clustering_with_genre_sets,
    find_best_k_by_elbow,
)
from data_collection import run_scrape
from util import (
    build_transition_network_by_item_cluster,
    build_transition_network_by_user_group,
    plot_embeddings_tsne,
    plot_with_umap,
    print_cluster_counts_and_ratios,
)
from word2vec import (  # noqa
    BookDataset,
    GridSearch,
    MovieDataset,
    OptunaSearch,
    Trainer,
)

# --- OMP関連のメッセージを抑制 ---
os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["KMP_WARNINGS"] = "0"
# --- Pythonの警告を抑制 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args_dict: Dict[str, Any]):
    # 　0. ハイパーパラメータのロード
    whole_args, word2vec_config, network_config = parse_config(args_dict)
    down_sample = word2vec_config.down_sample
    sample = word2vec_config.sample
    min_user_cnt = word2vec_config.min_user_cnt

    # 1. データの読み込み
    BASE_DIR = Path(__file__).resolve().parent
    if whole_args.dataset == "Book":
        DATA_PATH = BASE_DIR / "book_dataset" / "all_users_results.json"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        if not os.path.isfile(DATA_PATH):
            run_scrape()
        dataloader = BookDataset(
            folder_path=DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    else:
        DATA_PATH = BASE_DIR / "ml-1m"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        dataloader = MovieDataset(
            movie_dir=DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    dataloader.preprocess()

    # 2. word2vecのトレイナー読み込みと学習
    if whole_args.grid_search_flag:
        searcher = OptunaSearch(
            word2vec_config=word2vec_config,
            weight_path=WEIGHT_PATH,
            dataset=dataloader,
        )
        best_param, _ = searcher.search()
        # === best_param に基づき Trainer を再定義 ===
        trainer = Trainer(
            weight_path=WEIGHT_PATH,
            dataset=dataloader,
            embedding_dim=best_param["embedding_dim"],
            num_negatives=best_param["num_negatives"],
            learning_rate=best_param["learning_rate"],
            batch_size=word2vec_config.batch_size,
            epochs=word2vec_config.epochs,
            scheduler_factor=word2vec_config.scheduler_factor,
            early_stop_threshold=word2vec_config.early_stop_threshold,
            grid_search=False,  # 本番学習なので grid_search フラグはオフ
        )
    else:
        trainer = Trainer(
            weight_path=WEIGHT_PATH,
            dataset=dataloader,
            embedding_dim=word2vec_config.embedding_dim,
            num_negatives=word2vec_config.num_negatives,
            learning_rate=word2vec_config.learning_rate,
            batch_size=word2vec_config.batch_size,
            epochs=word2vec_config.epochs,
            scheduler_factor=word2vec_config.scheduler_factor,
            early_stop_threshold=word2vec_config.early_stop_threshold,
            grid_search=False,
        )

    if not os.path.isfile(WEIGHT_PATH) or whole_args.retrain:
        trainer.train()
    else:
        trainer._read_weight_vec(
            device="mps" if torch.backends.mps.is_available() else "cpu", weight_path=WEIGHT_PATH
        )

    # 3.行動(映画タイトル，本のタイトル)の特徴量を用いたクラスタリング
    vec_df = trainer.vec_df
    if whole_args.dataset == "Movie":
        # メタデータの整形
        meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
        meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])

        cluster_genre_dist, lrap, labels = evaluate_clustering_with_genre_sets(
            vec_df,
            meta_df,
            id_col="title",
            genre_col="genres",
            save_dir=BASE_DIR,
        )
        item_cluster_labels = dict(zip(vec_df.index, labels))
        print_cluster_counts_and_ratios(item_cluster_labels)
    else:
        # 通常のクラスタリング（ジャンル評価なし）
        optimal_k = find_best_k_by_elbow(vec_df.values, max_k=20)
        kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(vec_df.values)
        item_cluster_labels = dict(zip(vec_df.index, labels))
        print_cluster_counts_and_ratios(item_cluster_labels)

    plot_embeddings_tsne(
        embeddings=trainer.book_embeddings,
        labels=labels,
        label_name="artwork",
        save_dir=BASE_DIR,
    )
    plot_with_umap(
        embeddings=trainer.book_embeddings, labels=labels, label_name="artwork", umap_dir=BASE_DIR
    )

    build_transition_network_by_item_cluster(
        data_df=dataloader.data_gen,
        item_cluster_labels=item_cluster_labels,
        save_dir=BASE_DIR,
        meta_df=meta_df,
        min_weight=network_config.item_weight,
    )

    # 4.word2vecの結果を用いたユーザーのクラスタリング
    user_embeddings = trainer.user_embeddings
    user_df = dataloader.data_gen[["userId", "gender"]].drop_duplicates()
    assert user_embeddings.shape[0] == len(user_df), "ユーザー数が一致しません"

    n_user_clusters = find_best_k_by_elbow(user_embeddings, max_k=20)
    print(f"[INFO] Elbow法により選ばれたクラスタ数: {n_user_clusters}")

    kmeans = KMeans(n_clusters=n_user_clusters, n_init="auto", random_state=42)
    user_cluster_labels = kmeans.fit_predict(user_embeddings)
    user_df["cluster"] = user_cluster_labels

    plot_embeddings_tsne(
        embeddings=trainer.user_embeddings,
        labels=user_cluster_labels,
        label_name="user",
        save_dir=BASE_DIR,
    )

    # 5.  ユーザークラスタごとのネットワーク作成
    PLT_RESULT_DIR = BASE_DIR / "plt" / "network"
    total_users = int(dataloader.data_gen["userId"].nunique())  # user_idに統一した方がええかも
    cluster_groups = {
        f"user_cluster_{cid}": dataloader.data_gen["userId"].isin(
            user_df[user_df["cluster"] == cid]["userId"]
        )
        for cid in range(n_user_clusters)
    }

    cluster_user_weight = network_config.cluster_user_weight
    for group_label, group_filter in cluster_groups.items():
        build_transition_network_by_user_group(
            data_df=dataloader.data_gen,
            group_filter=group_filter,
            group_label=group_label,
            item_cluster_labels=item_cluster_labels,
            save_dir=PLT_RESULT_DIR / "clustered_users",
            meta_df=meta_df,
            base_weight=cluster_user_weight,
            total_users=total_users,
        )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
