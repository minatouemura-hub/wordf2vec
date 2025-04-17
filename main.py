import argparse  # noqa
import os  # noqa
import sys  # noqa

# 上記設定を行ったあとにnumpyやscipyなどをインポート
import warnings
from collections import Counter, defaultdict  # noqa
from pathlib import Path
from typing import Any, Dict

import numpy as np  # noqa
import pandas as pd  # Noqa; noqa
import torch
from grakel import Graph, GraphKernel
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances  # noqa
from tqdm import tqdm

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


def build_user_graphs(data_df):
    user_graphs = []
    user_ids = []
    for user_id, group in tqdm(data_df.groupby("userId"), desc="ユーザーグラフの構築中"):
        edges = []
        node_labels = {}
        titles = group.sort_values("timestamp")["title"].tolist()
        for idx, (u, v) in enumerate(zip(titles, titles[1:])):
            edges.append((u, v))
            node_labels[u] = u  # ラベルにタイトル名（またはユニークなID）を入れる
            node_labels[v] = v
        if edges:
            user_graphs.append(Graph(edges, node_labels=node_labels))
            user_ids.append(user_id)
    return user_graphs, user_ids


def compute_graph_kernel_matrix(graphs):
    gk = GraphKernel(kernel=["weisfeiler_lehman", "shortest_path"], normalize=True)
    return gk.fit_transform(graphs)


def cluster_users_by_graph_kernel(K, n_clusters=10):
    model = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
    return model.fit_predict(K)


def main(args_dict: Dict[str, Any]):
    whole_args, word2vec_config, network_config = parse_config(args_dict)
    down_sample = word2vec_config.down_sample
    sample = word2vec_config.sample
    min_user_cnt = word2vec_config.min_user_cnt

    BASE_DIR = Path(__file__).resolve().parent
    if whole_args.dataset == "Book":
        DATA_PATH = BASE_DIR / "book_dataset" / "all_users_results.json"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        if not os.path.isfile(DATA_PATH):
            run_scrape()
        dataloader = BookDataset(
            DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    else:
        DATA_PATH = BASE_DIR / "ml-1m"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        dataloader = MovieDataset(
            DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    dataloader.preprocess()

    if whole_args.grid_search_flag:
        searcher = OptunaSearch(word2vec_config, WEIGHT_PATH, dataloader)
        best_param, _ = searcher.search()
        trainer = Trainer(
            WEIGHT_PATH,
            dataset=dataloader,
            embedding_dim=best_param["embedding_dim"],
            num_negatives=best_param["num_negatives"],
            learning_rate=best_param["learning_rate"],
            batch_size=word2vec_config.batch_size,
            epochs=word2vec_config.epochs,
            scheduler_factor=word2vec_config.scheduler_factor,
            early_stop_threshold=word2vec_config.early_stop_threshold,
        )
    else:
        trainer = Trainer(
            WEIGHT_PATH,
            dataset=dataloader,
            embedding_dim=word2vec_config.embedding_dim,
            num_negatives=word2vec_config.num_negatives,
            learning_rate=word2vec_config.learning_rate,
            batch_size=word2vec_config.batch_size,
            epochs=word2vec_config.epochs,
            scheduler_factor=word2vec_config.scheduler_factor,
            early_stop_threshold=word2vec_config.early_stop_threshold,
        )

    if not os.path.isfile(WEIGHT_PATH) or whole_args.retrain:
        trainer.train()
    else:
        trainer._read_weight_vec(
            device="cuda" if torch.cuda.is_available() else "cpu", weight_path=WEIGHT_PATH
        )

    vec_df = trainer.vec_df
    if whole_args.dataset == "Movie":
        meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
        meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])
        _, _, labels = evaluate_clustering_with_genre_sets(
            vec_df, meta_df, "title", "genres", BASE_DIR
        )
        item_cluster_labels = dict(zip(vec_df.index, labels))
    else:
        optimal_k = find_best_k_by_elbow(vec_df.values, max_k=20)
        kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(vec_df.values)
        item_cluster_labels = dict(zip(vec_df.index, labels))

    print_cluster_counts_and_ratios(item_cluster_labels)
    plot_embeddings_tsne(trainer.book_embeddings, BASE_DIR, labels, "artwork")
    plot_with_umap(trainer.book_embeddings, labels, "artwork", BASE_DIR)

    build_transition_network_by_item_cluster(
        dataloader.data_gen, item_cluster_labels, BASE_DIR, meta_df, network_config.item_weight
    )

    # ==== Graph Kernel によるユーザークラスタリング ====
    user_graphs, user_ids = build_user_graphs(dataloader.data_gen)
    K = compute_graph_kernel_matrix(user_graphs)
    user_cluster_labels = cluster_users_by_graph_kernel(K, n_clusters=10)

    user_df = dataloader.data_gen[["userId", "gender"]].drop_duplicates()
    user_df = user_df[user_df["userId"].isin(user_ids)].copy()
    user_df["cluster"] = user_cluster_labels

    # ==== ハイブリッドネットワーク作成 ====
    PLT_RESULT_DIR = BASE_DIR / "plt" / "network"
    total_users = dataloader.data_gen["userId"].nunique()
    title2idx = {title: idx for idx, title in enumerate(vec_df.index)}

    cluster_groups = {
        f"user_cluster_{cid}": dataloader.data_gen["userId"].isin(
            user_df[user_df["cluster"] == cid]["userId"]
        )
        for cid in np.unique(user_cluster_labels)
    }

    for group_label, group_filter in cluster_groups.items():
        build_transition_network_by_user_group(
            data_df=dataloader.data_gen,
            group_filter=group_filter,
            group_label=group_label,
            item_cluster_labels=item_cluster_labels,
            save_dir=PLT_RESULT_DIR / "clustered_users",
            meta_df=meta_df,
            embeddings=trainer.book_embeddings,
            title2idx=title2idx,
            base_weight=network_config.cluster_user_weight,
            total_users=total_users,
        )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
