import argparse  # noqa
import os  # noqa
import sys  # noqa
from pathlib import Path
from typing import Any, Dict

import pandas as pd  # noqa
import torch

from arg import get_args, parse_config
from cluster_analysis.analysis import evaluate_clustering_with_genre_sets
from data_collection import run_scrape
from gender_axis.projection import Project_On  # noqa
from util import (
    build_transition_network_by_item_cluster,
    build_transition_network_by_user_group,
    print_cluster_counts_and_ratios,
)
from word2vec import Trainer  # noqa
from word2vec import BookDataset, MovieDataset  # noqa


def main(args_dict: Dict[str, Any]):
    # 諸設定の読み込み
    whole_args, projection_args, word2vec_config = parse_config(args_dict)
    down_sample = word2vec_config.down_sample
    sample = word2vec_config.sample
    min_user_cnt = word2vec_config.min_user_cnt

    BASE_DIR = Path(__file__).resolve().parent  # noqa
    if whole_args.dataset == "Book":
        DATA_PATH = BASE_DIR / "data" / "all_users_results.json"
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

    # 2. word2vecの訓練開始
    trainer = Trainer(
        DATA_PATH,
        WEIGHT_PATH,
        dataset=dataloader,
        embedding_dim=word2vec_config.embedding_dim,
        num_negatives=word2vec_config.num_negatives,
        learning_rate=word2vec_config.learning_rate,
        batch_size=word2vec_config.batch_size,
        epochs=word2vec_config.epochs,
        scheduler_factor=word2vec_config.scheduler_factor,
        early_stop_threshold=word2vec_config.early_stop_threshold,
        top_range=word2vec_config.top_range,
        eval_task=word2vec_config.task_name,
    )

    if not os.path.isfile(WEIGHT_PATH) or whole_args.retrain:
        trainer.train()
    else:
        trainer._read_weight_vec(
            device="mps" if torch.backends.mps.is_available() else "cpu", weight_path=WEIGHT_PATH
        )

    # 3. word2vecの評価
    vec_df = trainer.vec_df  # index = title
    meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
    meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])

    # 4. アイテムクラスタの確認（必要に応じて実施）
    _, _, labels = evaluate_clustering_with_genre_sets(
        vec_df=vec_df, meta_df=meta_df, id_col="title", n_clusters=20, save_dir=BASE_DIR
    )
    item_cluster_labels = dict(zip(vec_df.index, labels))
    print_cluster_counts_and_ratios(item_cluster_labels)

    build_transition_network_by_item_cluster(
        data_df=dataloader.data_gen,
        item_cluster_labels=item_cluster_labels,
        save_dir=BASE_DIR,
        meta_df=meta_df,
        min_weight=50,
    )

    # 5. ユーザー属性（gender × age）に基づくネットワーク作成
    groups = {
        "adult_female": (dataloader.data_gen["gender"] == "F") & (dataloader.data_gen["age"] >= 20),
        "adult_male": (dataloader.data_gen["gender"] == "M") & (dataloader.data_gen["age"] >= 20),
        "minor_female": (dataloader.data_gen["gender"] == "F") & (dataloader.data_gen["age"] < 20),
        "minor_male": (dataloader.data_gen["gender"] == "M") & (dataloader.data_gen["age"] < 20),
    }
    for group_label, group_filter in groups.items():
        build_transition_network_by_user_group(
            data_df=dataloader.data_gen,
            group_filter=group_filter,
            group_label=group_label,
            item_cluster_labels=item_cluster_labels,
            save_dir=BASE_DIR,
            meta_df=meta_df,
            min_weight=20,
        )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
