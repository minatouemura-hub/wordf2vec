import argparse  # noqa
import os  # noqa
import sys  # noqa

# 上記設定を行ったあとにnumpyやscipyなどをインポート
import warnings
from collections import Counter, defaultdict  # noqa
from pathlib import Path
from typing import Any, Dict

import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances  # noqa

from arg import get_args, parse_config
from cluster_analysis.analysis import Balanced_Kmeans  # noqa
from cluster_analysis.analysis import (
    evaluate_clustering_with_genre_sets,
    evaluate_fast_greedy_with_genre_sets,
    find_best_k_by_elbow,
)
from data_collection import run_scrape
from util import build_transition_network_by_user_group  # noqa
from util import (
    analyze_cluster_transitions,
    build_transition_network_by_item_cluster,
    fast_greedy_clustering_from_network,
    plot_embeddings_tsne,
    plot_rating_over_exposures,
    plot_with_umap,
    print_cluster_counts_and_ratios,
)
from word2vec import (  # noqa
    BookDataset,
    GridSearch,
    Movie1MDataset,
    Movie10MDataset,
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
    elif whole_args.dataset == "Movie1M":
        DATA_PATH = BASE_DIR / "ml-1m"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        dataloader = Movie1MDataset(
            DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    else:
        DATA_PATH = BASE_DIR / "ml-10m"
        WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
        dataloader = Movie10MDataset(
            DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    dataloader.preprocess()

    if whole_args.grid_search_flag:
        searcher = OptunaSearch(
            word2vec_config=word2vec_config, weight_path=WEIGHT_PATH, dataset=dataloader
        )
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
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        trainer._read_weight_vec(device=device, weight_path=WEIGHT_PATH)

    vec_df = trainer.vec_df
    if whole_args.dataset != "Book":
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

    if whole_args.fast_greedy_compare:
        fg_cluster_labels, fg_graph = fast_greedy_clustering_from_network(dataloader.data_gen)
        if meta_df is not None:
            evaluate_fast_greedy_with_genre_sets(
                item_cluster_labels=fg_cluster_labels, meta_df=meta_df, save_dir=BASE_DIR
            )
    analyze_cluster_transitions(
        data_df=dataloader.data_gen, item_cluster_labels=item_cluster_labels, save_dir=BASE_DIR
    )

    # main関数内の最後で呼び出し
    plot_rating_over_exposures(
        data_df=dataloader.data_gen,
        item_cluster_labels=item_cluster_labels,
        save_dir=BASE_DIR,
        min_cluster_samples=50,
    )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
