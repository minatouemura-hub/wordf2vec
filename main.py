import argparse  # noqa
import os  # noqa
import sys  # noqa
from pathlib import Path
from typing import Any, Dict

import pandas as pd  # noqa
import torch

from arg import get_args, parse_config
from cluster_analysis.analysis import (
    compare_cluster_entropy_by_gender,
    evaluate_clustering_with_genre_sets,
)
from data_collection import run_scrape
from gender_axis.projection import Project_On  # noqa
from word2vec import Trainer  # noqa
from word2vec import BookDataset, MovieDataset  # noqa


def main(args_dict: Dict[str, Any]):
    whole_args, projection_args, word2vec_config = parse_config(args_dict)
    down_sample = word2vec_config.down_sample
    sample = word2vec_config.sample
    min_user_cnt = word2vec_config.min_user_cnt

    # 1. 各種設定とデータのロード
    BASE_DIR = Path(__file__).resolve().parent  # noqa
    if whole_args.dataset == "Book":
        DATA_PATH = BASE_DIR / "data" / "all_users_results.json"
        if not os.path.isfile(DATA_PATH):
            run_scrape()
        dataloader = BookDataset(
            folder_path=DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
        )
    DATA_PATH = BASE_DIR / "ml-1m"
    WEIGHT_PATH = BASE_DIR / "weight_vec" / f"{whole_args.dataset}2vec_model.pth"
    dataloader = MovieDataset(
        movie_dir=DATA_PATH, down_sample=down_sample, sample=sample, min_user_cnt=min_user_cnt
    )
    dataloader.preprocess()

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
    # 2. 訓練の実行
    if not os.path.isfile(WEIGHT_PATH) or whole_args.retrain:
        trainer.train()
    else:
        trainer._read_weight_vec(
            device="mps" if torch.backends.mps.is_available() else "cpu", weight_path=WEIGHT_PATH
        )
        # 学習済みベクトルとメタデータを準備
    vec_df = trainer.vec_df  # index = title
    meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
    meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])

    # 3.評価フェーズ
    # word2vecの結果表示=> 既存の映画分類と一致している．意味空間がうまく学習されている
    _, _, cluster_labels = evaluate_clustering_with_genre_sets(
        vec_df=vec_df,
        meta_df=meta_df,
        id_col="title",
        n_clusters=20,  # grid searchしたい..　あとpathを通してくれ
    )

    # 4.分析フェーズ
    # - 1次元に射影してユーザーの分析
    # - 先のクラスタを利用して，男女の読書傾向が見られるか(女性の方が幅広い等)
    # - 性別可視化(結果は変わらずって感じ)
    compare_cluster_entropy_by_gender(
        data_df=dataloader.data_gen,
        cluster_labels=cluster_labels,
        id2book=dataloader.id2book,
        num_cluster=20,
        save_dir=BASE_DIR,
    )
    # -


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
