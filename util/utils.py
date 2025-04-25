import argparse  # noqa
import os  # noqa
import sys  # noqa
from collections import Counter, defaultdict  # noqa
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd  # noqa
import seaborn as sns
from matplotlib import colormaps
from matplotlib.patches import Patch
from networkx.algorithms.community import greedy_modularity_communities, modularity
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
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
    data_df, item_cluster_labels, save_dir, meta_df, min_weight=3
):
    cmap = colormaps.get_cmap("tab20")
    title_to_movieId = dict(zip(meta_df["title"], meta_df["movieId"]))

    cluster_to_titles = defaultdict(set)
    for title, cluster_id in item_cluster_labels.items():
        cluster_to_titles[cluster_id].add(title)

    for cluster_id, relevant_titles in tqdm(
        cluster_to_titles.items(), desc="アイテムネットワーク構築中"
    ):
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
            G.add_edge(src, dst, weight=weight)  # ← すべてのエッジを一旦追加

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue

        # === モジュラリティを全エッジで評価 ===
        ug = G.to_undirected()
        communities = greedy_modularity_communities(ug)
        mod_score = modularity(ug, communities)
        print(f"クラスタ {cluster_id} のモジュラリティ（全エッジ使用）: {mod_score:.4f}")

        # === 可視化対象のエッジ（min_weight以上）だけ残す ===
        filtered_G = nx.DiGraph()
        for (u, v), d in G.edges.items():
            if d["weight"] >= min_weight:
                filtered_G.add_edge(u, v, weight=d["weight"])

        if filtered_G.number_of_edges() == 0:
            continue

        centrality = nx.out_degree_centrality(filtered_G)
        node_colors = [cmap(item_cluster_labels.get(n, -1) % 20) for n in filtered_G.nodes()]
        node_sizes = [500 + 3000 * centrality.get(n, 0) for n in filtered_G.nodes()]

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(filtered_G, seed=42, k=1.2)
        weights = [filtered_G[u][v]["weight"] for u, v in filtered_G.edges()]
        max_weight = max(weights)
        edge_widths = [w * 0.1 for w in weights]
        edge_alphas = [0.2 + 0.8 * (w / max_weight) for w in weights]

        nx.draw_networkx_nodes(
            filtered_G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.6,
            edgecolors="white",
        )
        for (u, v), width, alpha in zip(filtered_G.edges(), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                filtered_G,
                pos,
                edgelist=[(u, v)],
                arrows=True,
                width=width,
                edge_color="gray",
                alpha=alpha,
            )

        label_nodes = {
            n: str(title_to_movieId[n])
            for n in filtered_G.nodes()
            if (500 + 1500 * centrality.get(n, 0)) > 700 and n in title_to_movieId
        }
        nx.draw_networkx_labels(
            filtered_G, pos, labels=label_nodes, font_size=8, font_color="black"
        )

        unique_clusters = set(item_cluster_labels.get(n, -1) for n in filtered_G.nodes())
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


def build_transition_network_by_user_group(
    data_df,
    group_filter,
    group_label,
    save_dir,
    meta_df,
    item_cluster_labels,
    embeddings: np.ndarray,
    title2idx: dict,
    base_weight: float,
    total_users: int,
    k_nn: int = 5,
    alpha: float = 0.7,
    edge_weight_threshold: float = 1.0,  # 描画対象のエッジ重みの閾値
):
    """
    ユーザー行動遷移グラフと埋め込み類似度による KNN グラフを合成し、
    可視化時には一定以上の重みを持つエッジのみを残す。
    """
    cmap = colormaps.get_cmap("tab20")

    # 1) 埋め込み距離に基づく KNN グラフを構築
    titles = list(meta_df["title"])
    dist_mat = pairwise_distances(embeddings, metric="euclidean")
    G_emb = nx.DiGraph()
    for title in titles:
        i = title2idx.get(title)
        if i is None:
            continue
        neigh = np.argsort(dist_mat[i])[1 : k_nn + 1]
        for j in neigh:
            tgt = titles[j]
            d = dist_mat[i, j]
            if np.isfinite(d):
                G_emb.add_edge(title, tgt, weight=1.0 / (d + 1e-6))

    # 2) 行動遷移グラフを構築
    filtered_df = data_df[group_filter].copy()
    filtered_user_num = int(filtered_df["userId"].nunique())
    adjusted_w = max(int(filtered_user_num / total_users * base_weight), 1)

    edge_counter = defaultdict(int)
    for _, grp in filtered_df.groupby("userId"):
        actions = grp.sort_values("timestamp")["title"].tolist()
        for u, v in zip(actions, actions[1:]):
            edge_counter[(u, v)] += 1

    G = nx.DiGraph()
    for (u, v), w in edge_counter.items():
        if w >= adjusted_w:
            G.add_edge(u, v, weight=alpha * w)

    # 3) 埋め込み KNN グラフのエッジをマージ
    for u, v, d in G_emb.edges(data=True):
        emb_w = (1 - alpha) * d["weight"]
        if G.has_edge(u, v):
            G[u][v]["weight"] += emb_w
        else:
            G.add_edge(u, v, weight=emb_w)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print(f"グループ {group_label} のネットワークが構築できませんでした。")
        return

    # モジュラリティ（全体グラフで評価）
    # ug = G.to_undirected()
    # comms = greedy_modularity_communities(ug)
    # mod_val = modularity(ug, comms)

    # 可視化：閾値以上の重みを持つエッジのみ抽出
    important_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] >= edge_weight_threshold]
    drawn_nodes = set([u for u, v in important_edges] + [v for u, v in important_edges])

    if not drawn_nodes:
        print(f"グループ {group_label} に表示可能なノードがありません。")
        return

    # 可視ノードに対応する中心性
    centrality = nx.in_degree_centrality(G)
    node_colors = [cmap(item_cluster_labels.get(n, -1) % 20) for n in drawn_nodes]
    node_sizes = [200 + 5000 * centrality.get(n, 0) for n in drawn_nodes]

    # レイアウト・描画
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G)
    weights = [G[u][v]["weight"] for u, v in important_edges]
    max_w = max(weights) if weights else 1
    widths = [0.2 + 2.8 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=drawn_nodes,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.6,
        edgecolors="white",
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=important_edges, width=widths, arrowstyle="->", arrowsize=8, alpha=0.4
    )

    legend_elems = [
        Patch(facecolor=cmap(cid % 20), edgecolor="black", label=f"クラスタ{cid}")
        for cid in sorted(set(item_cluster_labels.get(n, -1) for n in drawn_nodes))
    ]
    plt.legend(
        handles=legend_elems,
        title="ノード所属クラスタ",
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    plt.title(f"ユーザーグループ別ハイブリッドネットワーク（{group_label}）")
    plt.axis("off")
    save_path = save_dir / f"{group_label}_hybrid_network.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"グループ {group_label} のハイブリッドネットワーク構築が完了しました。")


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


def fast_greedy_clustering_from_network(data_df: pd.DataFrame):
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("community (python-louvain) module not found. Please install it via pip.")

    item_graph = nx.Graph()
    transitions = data_df.sort_values("timestamp").groupby("userId")["title"].apply(list)
    for title_list in transitions:
        for u, v in zip(title_list, title_list[1:]):
            item_graph.add_edge(u, v)

    partition = community_louvain.best_partition(item_graph)
    return dict(partition), item_graph


def analyze_cluster_transitions(data_df, item_cluster_labels, save_dir):

    transition_counts = defaultdict(Counter)
    return_step_counts = defaultdict(list)
    total_visits = Counter()
    unique_return_users = defaultdict(set)

    # 各ユーザーの行動を時系列順に取得
    for user_id, group in data_df.groupby("userId"):
        actions = group.sort_values("timestamp")["title"].tolist()
        clusters = [
            item_cluster_labels.get(title, -1) for title in actions if title in item_cluster_labels
        ]

        prev_cluster = None
        last_seen = {}
        for step, curr_cluster in enumerate(clusters):
            total_visits[curr_cluster] += 1
            if prev_cluster is not None and prev_cluster != curr_cluster:
                transition_counts[prev_cluster][curr_cluster] += 1
                if curr_cluster in last_seen:
                    return_step = step - last_seen[curr_cluster]
                    if return_step > 1:  # 即時回帰を除外
                        return_step_counts[curr_cluster].append(return_step)
                        unique_return_users[curr_cluster].add(user_id)
            last_seen[curr_cluster] = step
            prev_cluster = curr_cluster

    # 遷移確率行列の可視化
    cluster_ids = sorted(set(item_cluster_labels.values()))
    matrix = pd.DataFrame(index=cluster_ids, columns=cluster_ids).fillna(0.0)
    for src in transition_counts:
        total = sum(transition_counts[src].values())
        for tgt in transition_counts[src]:
            matrix.loc[src, tgt] = transition_counts[src][tgt] / total

    save_path = Path(save_dir) / "plt" / "cluster_transition"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Cluster-to-Cluster Transition Probability")
    plt.xlabel("To Cluster")
    plt.ylabel("From Cluster")
    plt.savefig(save_path / "transition_matrix.png")
    plt.close()

    # 戻りステップ数の分布と統計指標の保存
    stats = []
    for cluster_id, steps in return_step_counts.items():
        if steps:
            plt.figure()
            sns.histplot(steps, bins=20, kde=True)
            plt.title(f"Return Step Distribution to Cluster {cluster_id}")
            plt.xlabel("Steps Until Return")
            plt.ylabel("Frequency")
            plt.savefig(save_path / f"return_steps_cluster_{cluster_id}.png")
            plt.close()

            avg_step = np.mean(steps)
            return_rate = len(unique_return_users[cluster_id]) / total_visits[cluster_id]
            stats.append((cluster_id, avg_step, return_rate))

    if stats:
        stat_df = pd.DataFrame(stats, columns=["Cluster", "AvgReturnStep", "ReturnRate"])
        stat_df.to_csv(save_path / "return_stats.csv", index=False)
        print("[INFO] 平均戻りステップ数と回帰率を保存しました。")

    print("[INFO] クラスタ間遷移と戻りステップの分析が完了しました。")


def plot_rating_over_exposures(
    data_df, item_cluster_labels, save_dir, min_cluster_samples: int = 50
):
    """
    MEE 検証用プロット生成関数
    1) rating をユーザ z-score 正規化
    2) 全クラスタ統合の MEE 曲線を描画
    3) sample 数が min_cluster_samples 未満のクラスタを除外し
       各クラスタ別 MEE 曲線を補助的に描画
    """

    base_path = Path(save_dir) / "plt" / "mee_plot"
    base_path.mkdir(parents=True, exist_ok=True)
    cluster_path = base_path / "by_cluster"
    cluster_path.mkdir(parents=True, exist_ok=True)

    # --- ① rating のユーザ基準化（z-score） ------------------------------
    data_df = data_df.copy()
    user_stats = data_df.groupby("userId")["rating"].agg(["mean", "std"]).reset_index()
    user_stats["std"].replace(0, 1e-6, inplace=True)  # 分散ゼロ対策
    data_df = data_df.merge(user_stats, on="userId", how="left")
    data_df["rating_z"] = (data_df["rating"] - data_df["mean"]) / data_df["std"]

    # --- ② クラスタ情報付与と exposure 計算 ------------------------------
    data_df["cluster"] = data_df["title"].map(item_cluster_labels)
    data_df.dropna(subset=["cluster"], inplace=True)
    data_df = data_df.sort_values(["userId", "timestamp"])
    data_df["cluster_exposure"] = data_df.groupby(["userId", "cluster"]).cumcount() + 1

    # pair 単位の総 exposure
    repetition_counts = data_df.groupby(["userId", "cluster"]).size().reset_index(name="total_reps")
    data_df = data_df.merge(repetition_counts, on=["userId", "cluster"])

    # 5〜50回にトリミング
    data_df = data_df[(data_df["total_reps"] >= 5) & (data_df["total_reps"] <= 50)]

    # rep クラス付与
    def categorize(rep):
        return (
            "LowRep"
            if rep <= 16
            else "ModRep" if rep <= 27 else "HighRep" if rep <= 38 else "VHRep"
        )

    data_df["rep_class"] = data_df["total_reps"].apply(categorize)

    # ===== ③ 統合プロット =================================================
    plot_df_all = (
        data_df.groupby(["rep_class", "cluster_exposure"])["rating_z"].mean().reset_index()
    )
    plt.figure(figsize=(10, 6))
    for cls in ["LowRep", "ModRep", "HighRep", "VHRep"]:
        cls_df = plot_df_all[plot_df_all["rep_class"] == cls]
        plt.plot(cls_df["cluster_exposure"], cls_df["rating_z"], label=cls, marker="o")
    plt.title("Average *Z-scored* Rating over Cluster Exposure (All Clusters)")
    plt.xlabel("Exposure Count to Same Cluster")
    plt.ylabel("Average Z-scored Rating")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path / "cluster_based_mee_plot.png")
    plt.close()

    # ===== ④ クラスタ別プロット ==========================================
    cluster_counts = data_df["cluster"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cluster_samples].index
    plot_df_by_cluster = (
        data_df[data_df["cluster"].isin(valid_clusters)]
        .groupby(["cluster", "rep_class", "cluster_exposure"])["rating_z"]
        .mean()
        .reset_index()
    )
    for cluster_id in sorted(valid_clusters):
        cdf = plot_df_by_cluster[plot_df_by_cluster["cluster"] == cluster_id]
        plt.figure(figsize=(10, 6))
        for cls in ["LowRep", "ModRep", "HighRep", "VHRep"]:
            tmp = cdf[cdf["rep_class"] == cls]
            if not tmp.empty:
                plt.plot(tmp["cluster_exposure"], tmp["rating_z"], label=cls, marker="o")
        plt.title(f"Cluster {cluster_id}: Z-scored Rating over Exposure (n≥{min_cluster_samples})")
        plt.xlabel("Exposure Count to Same Cluster")
        plt.ylabel("Average Z-scored Rating")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cluster_path / f"cluster_{cluster_id}_mee_plot.png")
        plt.close()

    print(
        f"[INFO] MEE プロット（ユーザz-score & n≥{min_cluster_samples}クラスタ）保存先: {base_path}"
    )


def main():
    return


if __name__ == "__main__":
    main()
