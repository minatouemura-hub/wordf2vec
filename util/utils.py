import argparse  # noqa
import os  # noqa
import sys  # noqa
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd  # noqa
from matplotlib import colormaps
from matplotlib.patches import Patch
from networkx.algorithms.community import greedy_modularity_communities, modularity
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP  # noqa


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
    data_df, item_cluster_labels, save_dir, meta_df, min_weight=50
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
        node_sizes = [500 + 3000 * centrality.get(n, 0) for n in G.nodes()]

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=1.2)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_weight = max(weights)
        edge_widths = [w * 0.1 for w in weights]
        edge_alphas = [0.2 + 0.8 * (w / max_weight) for w in weights]

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, edgecolors="white"
        )
        for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], arrows=True, width=width, edge_color="gray", alpha=alpha
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
        plt_path = save_dir / "plt" / "network" / "item_network"
        os.makedirs(plt_path, exist_ok=True)
        plt.savefig(plt_path / f"item_transition_network_cluster_{cluster_id}.png")
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


def build_transition_network_by_user_group(
    data_df,
    group_filter,
    group_label,
    save_dir,
    meta_df,
    item_cluster_labels,
    base_weight,
    total_users,
):
    cmap = colormaps.get_cmap("tab20")
    # ユーザー属性でフィルタリング
    filtered_df = data_df[group_filter].copy()
    filtered_user_num = int(filtered_df["userId"].nunique())
    adjusted_weight = int((filtered_user_num / total_users) * base_weight)
    adjusted_weight = adjusted_weight if adjusted_weight >= 1 else 1
    print(f"グループ {group_label} のユーザー数: {filtered_user_num}")

    # 各ユーザーの行動（タイトルの連続）からエッジをカウントする
    edge_counter = defaultdict(int)
    for user_id, group in filtered_df.groupby("userId"):
        group = group.sort_values("timestamp")
        actions = group["title"].tolist()
        for i in range(len(actions) - 1):
            src, dst = actions[i], actions[i + 1]
            edge_counter[(src, dst)] += 1

    # adjusted_weight以上のエッジのみを抽出してグラフへ追加
    G = nx.DiGraph()
    for (src, dst), weight in edge_counter.items():
        if weight >= adjusted_weight:
            G.add_edge(src, dst, weight=weight)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print(f"グループ {group_label} のネットワークが構築できませんでした。")
        return

    # モジュラリティ計算
    undirected_G = G.to_undirected()
    communities = greedy_modularity_communities(undirected_G)
    mod_value = modularity(undirected_G, communities)
    print(f"グループ {group_label} のモジュラリティ: {mod_value:.4f}")

    # 出次数中心性を計算（ノードサイズの決定に利用）
    centrality = nx.out_degree_centrality(G)
    node_colors = [cmap(item_cluster_labels.get(n, -1) % 20) for n in G.nodes()]
    node_sizes = [500 + 3000 * centrality.get(n, 0) for n in G.nodes()]

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=1.2)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [w * 0.1 for w in weights]
    edge_alphas = [0.2 + 0.8 * (w / max_weight) for w in weights]

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, edgecolors="white"
    )
    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], arrows=True, width=width, edge_color="gray", alpha=alpha
        )

    # meta情報（タイトル -> movieId）の辞書を作成
    title_to_movieId = dict(zip(meta_df["title"], meta_df["movieId"]))  # noqa
    # 各映画の平均評価を data_df から計算（rating列が存在する前提）
    avg_ratings = data_df.groupby("title")["rating"].mean().to_dict()

    # ノード中心に平均評価を描画
    for n in G.nodes():
        if n in avg_ratings:
            plt.text(
                pos[n][0],
                pos[n][1],
                f"{avg_ratings[n]:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                color="black",
            )

    unique_clusters = set(item_cluster_labels.get(n, -1) for n in G.nodes())
    legend_elements = [
        Patch(facecolor=cmap(cid % 20), edgecolor="black", label=f"クラスタ {cid}")
        for cid in sorted(unique_clusters)
    ]
    plt.legend(
        handles=legend_elements,
        title="アイテムの所属クラスタ",
        loc="upper right",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    plt.title(f"ユーザー属性グループ別アイテム遷移ネットワーク（{group_label}）")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    save_path = save_dir / f"{group_label}_action_network.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # centrality_df = pd.DataFrame.from_dict(
    #     centrality, orient="index", columns=["out_degree_centrality"]
    # )
    # centrality_path = (
    #     save_dir / "result" / "network" / f"{group_label}_action_network_centrality.csv"
    # )
    # centrality_path.parent.mkdir(parents=True, exist_ok=True)
    # centrality_df.to_csv(centrality_path)

    print(f"グループ {group_label} のネットワーク構築が完了しました。")


def print_cluster_counts_and_ratios(item_cluster_labels: dict):
    """
    各クラスタに属する映画タイトルの数と全体に対する割合を表示する関数。

    Parameters:
      item_cluster_labels: {タイトル: クラスタID} の辞書
    """
    cluster_counts = Counter(item_cluster_labels.values())
    total = sum(cluster_counts.values())
    print("各クラスタのタイトル数と全体に対する割合:")
    for cluster, count in sorted(cluster_counts.items()):
        ratio = count / total * 100
        print(f"  クラスタ {cluster}: {count} 件, {ratio:.1f}%")


def plot_embeddings_tsne(embeddings, save_dir: Path, labels=None, label_name="embedding"):
    """
    任意の埋め込みベクトルをt-SNEで2次元に圧縮し、可視化する。

    Parameters:
        embeddings (ndarray): shape=(n_samples, dim)
        save_dir (Path): 結果の画像保存先ディレクトリ
        labels (array-like): 各サンプルのラベル（クラスタなど）。色分けに使用（任意）
        label_name (str): プロットのタイトルやファイル名に使う埋め込みの名前
    """
    # 保存ディレクトリの作成
    tsne_dir = save_dir / "plt"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    # スケーリングによる密度調整（optionalだけど有効）
    embeddings = StandardScaler().fit_transform(embeddings)

    # t-SNE 設定（初期化方法・繰り返し回数・perplexity を明示）
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", n_iter=2000)
    emb_2d = tsne.fit_transform(embeddings)

    # プロット
    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], s=5, alpha=0.6)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.6, color="blue")

    plt.axis("off")
    save_path = tsne_dir / f"{label_name}_tsne.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] t-SNEプロットを {save_path} に保存しました")


def plot_with_umap(embeddings, labels=None, label_name="embedding", umap_dir=None):
    reducer = UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.array(labels)
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], s=5, alpha=0.6)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5, alpha=0.6)

    plt.axis("off")
    umap_dir = umap_dir / "plt"
    umap_dir.mkdir(parents=True, exist_ok=True)
    save_path = umap_dir / f"{label_name}_umap.png"
    plt.savefig(save_path)
    plt.close()
