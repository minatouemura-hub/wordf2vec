import argparse  # noqa
import os  # noqa
import sys  # noqa
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd  # noqa
import torch
from matplotlib import colormaps
from matplotlib.patches import Patch
from tqdm import tqdm

from arg import get_args, parse_config
from cluster_analysis.analysis import compare_cluster_entropy_by_gender  # noqa
from cluster_analysis.analysis import (
    detect_user_change_points,
    evaluate_clustering_with_genre_sets,
)
from data_collection import run_scrape
from gender_axis.projection import Project_On  # noqa
from word2vec import Trainer  # noqa
from word2vec import BookDataset, MovieDataset  # noqa


def process_user(user_id, group, vec_dict):
    group = group.sort_values("timestamp")
    titles = group["title"].values
    vectors = [vec_dict[t] for t in titles if t in vec_dict]
    if len(vectors) < 3:
        return user_id, []
    vec_series = np.stack(vectors)
    cps = detect_user_change_points(vec_series, pen=5)
    return user_id, cps


def compute_echo_chamber_score(G, item_cluster_labels, target_cluster_id):
    cluster_nodes = {n for n, cid in item_cluster_labels.items() if cid == target_cluster_id}
    internal_edges = 0
    external_edges = 0
    for u, v in G.edges():
        if u in cluster_nodes and v in cluster_nodes:
            internal_edges += 1
        elif u in cluster_nodes and v not in cluster_nodes:
            external_edges += 1
    total = internal_edges + external_edges
    if total == 0:
        return 0.0, internal_edges, external_edges
    score = internal_edges / total
    return score, internal_edges, external_edges


def build_transition_network_by_item_cluster(
    data_df, item_cluster_labels, save_dir, meta_df, min_weight=20
):
    cmap = colormaps.get_cmap("tab20")
    title_to_movieId = dict(zip(meta_df["title"], meta_df["movieId"]))

    cluster_to_titles = defaultdict(set)
    for title, cluster_id in item_cluster_labels.items():
        cluster_to_titles[cluster_id].add(title)

    echo_scores = []

    for cluster_id, relevant_titles in tqdm(cluster_to_titles.items(), desc="ネットワーク構築中"):
        edge_counter = defaultdict(int)
        for user_id, group in data_df.groupby("userId"):
            group = group.sort_values("timestamp")
            actions = group["title"].tolist()
            for i in range(len(actions) - 1):
                src, dst = actions[i], actions[i + 1]
                if src in relevant_titles or dst in relevant_titles:
                    edge_counter[(src, dst)] += 1

        G = nx.DiGraph()
        for (src, dst), weight in edge_counter.items():
            if weight >= min_weight:
                G.add_edge(src, dst, weight=weight)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue

        centrality = nx.out_degree_centrality(G)
        node_colors = [cmap(item_cluster_labels.get(n, -1) % 20) for n in G.nodes()]
        node_sizes = [500 + 1500 * centrality.get(n, 0) for n in G.nodes()]

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=1.2)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, edgecolors="black"
        )
        nx.draw_networkx_edges(
            G, pos, arrows=True, width=[w * 0.1 for w in weights], edge_color="gray", alpha=0.4
        )

        label_nodes = {
            n: str(title_to_movieId[n])
            for n in G.nodes()
            if (500 + 1500 * centrality.get(n, 0)) > 700 and n in title_to_movieId
        }
        nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=8, font_color="black")

        unique_clusters = set(item_cluster_labels.get(n, -1) for n in G.nodes())
        legend_elements = [
            Patch(facecolor=cmap(cid % 20), edgecolor="black", label=f"クラスタ {cid}")
            for cid in sorted(unique_clusters)
        ]
        plt.legend(
            handles=legend_elements,
            title="ノードの所属クラスタ",
            loc="upper right",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )

        plt.title(f"アイテム遷移ネットワーク（クラスタ {cluster_id}）")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig(
            save_dir / "plt" / "network" / f"item_transition_network_cluster_{cluster_id}.png"
        )
        plt.close()

        pd.DataFrame.from_dict(
            centrality, orient="index", columns=["out_degree_centrality"]
        ).to_csv(
            save_dir
            / "result"
            / "network"
            / f"item_transition_network_cluster_{cluster_id}_centrality.csv"
        )

        score, internal, external = compute_echo_chamber_score(G, item_cluster_labels, cluster_id)
        echo_scores.append(
            {
                "cluster_id": cluster_id,
                "echo_chamber_score": score,
                "internal_edges": internal,
                "external_edges": external,
            }
        )

    pd.DataFrame(echo_scores).to_csv(
        save_dir / "result" / "network" / "echo_chamber_scores.csv", index=False
    )


def main(args_dict: Dict[str, Any]):
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

    if not os.path.isfile(WEIGHT_PATH) or whole_args.retrain:
        trainer.train()
    else:
        trainer._read_weight_vec(
            device="mps" if torch.backends.mps.is_available() else "cpu", weight_path=WEIGHT_PATH
        )

    vec_df = trainer.vec_df  # index = title
    meta_df = dataloader.data_gen[["movieId", "title", "genres"]].drop_duplicates()
    meta_df["main_genre"] = meta_df["genres"].apply(lambda g: g.split("|")[0])

    _, _, labels = evaluate_clustering_with_genre_sets(
        vec_df=vec_df, meta_df=meta_df, id_col="title", n_clusters=20, save_dir=BASE_DIR
    )
    item_cluster_labels = dict(zip(vec_df.index, labels))
    build_transition_network_by_item_cluster(
        data_df=dataloader.data_gen,
        item_cluster_labels=item_cluster_labels,
        save_dir=BASE_DIR,
        meta_df=meta_df,
    )


if __name__ == "__main__":
    args_dict = get_args()
    main(args_dict)
