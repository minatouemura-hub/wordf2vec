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
from cluster_analysis.analysis import evaluate_fast_greedy_with_genre_sets  # noqa
from cluster_analysis.analysis import (
    evaluate_clustering_with_genre_sets,
    find_best_k_by_elbow,
)
from data_collection import run_scrape
from util import (
    plot_embeddings_tsne,
    plot_rating_by_year_subplots,
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

# noqa

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
    if whole_args.dataset == "Movie1M" or "Movie10M":
        meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
        meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])
        _, _, labels = evaluate_clustering_with_genre_sets(
            vec_df=vec_df,
            meta_df=meta_df,
            id_col="title",
            genre_col="genres",
            save_dir=BASE_DIR,
            k_range=range(2, 40),
            method="kmeans",
        )
        item_cluster_labels = dict(zip(vec_df.index, labels))
    else:
        optimal_k = find_best_k_by_elbow(vec_df.values, max_k=20)
        kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(vec_df.values)
        item_cluster_labels = dict(zip(vec_df.index, labels))

    print_cluster_counts_and_ratios(item_cluster_labels)
    plot_embeddings_tsne(trainer.book_embeddings, BASE_DIR, labels, "artwork")
    # UMAP 可視化 ＋ シルエットスコア取得
    plot_with_umap(
        embeddings=trainer.book_embeddings,
        labels=labels,
        label_name="artwork",
        umap_dir=BASE_DIR,
        scaler=True,  # シルエットスコアを返す
    )

    # 低サンプルクラスタの削除
    cluster_item_counts = Counter(item_cluster_labels.values())
    valid_item_clusters = {cid for cid, cnt in cluster_item_counts.items() if cnt > 10}
    item_cluster_labels = {
        title: cid for title, cid in item_cluster_labels.items() if cid in valid_item_clusters
    }

    mee_df = plot_rating_by_year_subplots(  # noqa
        data_df=dataloader.data_gen, item_cluster_labels=item_cluster_labels, save_dir=BASE_DIR
    )

    # # --- 2) 逆U字型にならないユーザー×クラスタを検出 ---
    # non_iU_pairs = detect_non_invU_users(mee_df, min_exposures=20)
    # print(f"MEEに当てはまらなかったユーザー×クラスタは {len(non_iU_pairs)} 件です")

    # # --- MEEにならなかったクラスタを持つユーザの視聴多様性　--
    # df_all = dataloader.data_gen.copy()
    # df_all["cluster"] = df_all["title"].map(item_cluster_labels)

    # non_iU_user_ids = {user for user, _ in non_iU_pairs}
    # df_non_data = df_all[df_all["userId"].isin(non_iU_user_ids)].copy()

    # user2ncluster_all = df_all.dropna(subset=["cluster"]).groupby("userId")["cluster"].nunique()
    # user2ncluster_non = (
    #     df_non_data.dropna(subset=["cluster"]).groupby("userId")["cluster"].nunique()
    # )
    # print("All users:\n", user2ncluster_all.describe())
    # print("\nNon-inverseU users:\n", user2ncluster_non.describe())

    # # 5) ヒストグラムで視覚的に比較
    # plt.figure(figsize=(8, 4))
    # bins = range(1, max(user2ncluster_all.max(), user2ncluster_non.max()) + 2)

    # plt.hist(user2ncluster_all, bins=bins, alpha=0.6, label="All users")
    # plt.hist(user2ncluster_non, bins=bins, alpha=0.6, label="Non-inverseU users")

    # plt.xlabel("Number of unique clusters visited per user")
    # plt.ylabel("Number of users")
    # plt.title("クラスタ訪問多様性の比較")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(BASE_DIR / "plt" / "non_iU_cluster_transition")

    # # mee_df から一意の (userId,cluster) ペア数
    # df_non = pd.DataFrame(non_iU_pairs, columns=["userId", "cluster"])
    # non_counts = df_non["cluster"].value_counts().sort_index()
    # df_pairs = mee_df[["userId", "cluster"]].drop_duplicates()
    # total_counts = df_pairs["cluster"].value_counts().sort_index()

    # # 統計テーブルを作成
    # stats = (
    #     pd.DataFrame({"non_iU_pairs": non_counts, "total_pairs": total_counts})
    #     .fillna(0)
    #     .astype(int)
    # )
    # stats["non_iU_ratio"] = (stats["non_iU_pairs"] / stats["total_pairs"]).round(3)
    # stats = stats.sort_values("non_iU_ratio", ascending=False)

    # print("\n=== クラスタ別：非逆Uペア数／全ペア数／比率 ===")
    # print(stats.to_string())

    # non_iU_user_ids = {user for user, _ in non_iU_pairs}
    # non_iU_df = dataloader.data_gen[dataloader.data_gen["userId"].isin(non_iU_user_ids)]

    # # 4) クラスタ間遷移分析（非逆Uユーザーのみ）
    # analyze_cluster_transitions(
    #     data_df=non_iU_df,
    #     item_cluster_labels=item_cluster_labels,
    #     save_dir=BASE_DIR / "plt" / "non_iU_cluster_transition",
    # )
    # analyze_cluster_transitions(
    #     data_df=df_all,
    #     item_cluster_labels=item_cluster_labels,
    #     save_dir=BASE_DIR / "plt" / "all_user_trans",
    # )

    # build_cluster_transition_network_for_group(
    #     data_df=non_iU_df,
    #     item_cluster_labels=item_cluster_labels,
    #     save_dir=BASE_DIR / "plt" / "non_iU_cluster_transition",
    #     min_count=3,  # 例：遷移回数3回以上のみ可視化
    # )
    # build_cluster_transition_network_for_group(
    #     data_df=dataloader.data_gen,
    #     item_cluster_labels=item_cluster_labels,
    #     save_dir=BASE_DIR / "plt" / "all_user_trans",
    #     min_count=3,  # 例：遷移回数3回以上のみ可視化
    # )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
