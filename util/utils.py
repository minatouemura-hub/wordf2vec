import argparse  # noqa
import os  # noqa
import re
import sys  # noqa
from collections import Counter, defaultdict  # noqa
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd  # noqa
import seaborn as sns
import statsmodels.api as sm
from matplotlib import colormaps
from matplotlib.patches import Patch
from scipy.stats import sem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
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

        # # === モジュラリティを全エッジで評価 ===
        # ug = G.to_undirected()
        # communities = greedy_modularity_communities(ug)
        # mod_score = modularity(ug, communities)
        # print(f"クラスタ {cluster_id} のモジュラリティ（全エッジ使用）: {mod_score:.4f}")

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
    tsne = TSNE(
        n_components=2,
        perplexity=10,  # ← まずは 30,  then 10→50 と試す
        learning_rate=500,  # ← 中間値から
        early_exaggeration=20.0,  # ← デフォルト 12→20 で広がり確認
        init="pca",
        n_iter=1000,
        random_state=42,
    )
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


def plot_with_umap(
    embeddings,
    labels=None,
    label_name="embedding",
    umap_dir=None,
    scaler=True,
):
    """
    PCA+UMAPで可視化する関数

    Parameters:
        embeddings (ndarray): shape=(n_samples, dim)
        labels (array-like): クラスタラベルなど（色分けに使用、任意）
        label_name (str): ファイル名及びプロットタイトルに使用
        umap_dir (Path): 結果保存先の親ディレクトリ
        scaler (bool): PCA前にStandardScalerするか
        return_metrics (bool): シルエットスコアを計算して返すか

    Returns:
        silhouette (float, optional): return_metrics=Trueの場合のみ返却
    """
    # 前処理: スケーリング
    X = embeddings.copy()
    if scaler:
        X = StandardScaler().fit_transform(X)

    # グリッドサーチパラメータ
    n_neighbors_list = [15, 30, 50]
    min_dist_list = [0.001, 0.01, 0.05]

    best_params = None
    best_score = -1.0
    best_emb_2d = None

    # グリッド探索
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="cosine",
                spread=1.5,
                random_state=42,
            )
            emb_2d = reducer.fit_transform(X)
            if labels is not None:
                score = silhouette_score(emb_2d, labels, metric="euclidean")
                if score > best_score:
                    best_score = score
                    best_params = (n_neighbors, min_dist)
                    best_emb_2d = emb_2d

    if labels is not None:
        db_score = davies_bouldin_score(best_emb_2d, labels)
    print(f"shilhouette:{best_score:.3f},dc_score:{db_score:.3f}")
    # 最良パラメータで再プロット
    n_nb, m_dist = best_params
    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.array(labels)
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(best_emb_2d[idx, 0], best_emb_2d[idx, 1], s=5, alpha=0.6)
    else:
        plt.scatter(best_emb_2d[:, 0], best_emb_2d[:, 1], s=5, alpha=0.6)
    plt.axis("off")
    plt.title(f"UMAP ({label_name}) NN={n_nb} MD={m_dist} Sil={best_score:.3f}")

    # 保存
    path = Path(umap_dir) / "plt"
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / f"{label_name}_umap_best.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
                    return_step_counts[curr_cluster].append(return_step)
                    unique_return_users[curr_cluster].add(user_id)
            last_seen[curr_cluster] = step
            prev_cluster = curr_cluster

    # 遷移確率行列の可視化
    valid_clusters = sorted([cid for cid, cnt in total_visits.items() if cnt >= 30])

    # --- 遷移確率行列の可視化（行・列とも valid_clusters のみ） ---
    matrix = pd.DataFrame(index=valid_clusters, columns=valid_clusters).fillna(0.0)
    for src in transition_counts:
        if src not in valid_clusters:
            continue
        total = sum(
            transition_counts[src][tgt] for tgt in transition_counts[src] if tgt in valid_clusters
        )
        if total == 0:
            continue
        for tgt, cnt in transition_counts[src].items():
            if tgt in valid_clusters:
                matrix.loc[src, tgt] = cnt / total

    save_path = Path(save_dir) / "plt" / "cluster_transition"
    save_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    # annot=False にして数値を非表示
    sns.heatmap(
        matrix, annot=False, cmap="Blues", xticklabels=valid_clusters, yticklabels=valid_clusters
    )
    plt.title("Cluster-to-Cluster Transition Probability")
    plt.xlabel("To Cluster")
    plt.ylabel("From Cluster")
    plt.savefig(save_path / "transition_matrix.png")
    plt.close()
    stats = []
    for cluster_id, steps in return_step_counts.items():
        avg_step = np.mean(steps)
        return_rate = len(unique_return_users[cluster_id]) / total_visits[cluster_id]
        stats.append((cluster_id, avg_step, return_rate))
    if stats:
        stat_df = pd.DataFrame(stats, columns=["Cluster", "AvgReturnStep", "ReturnRate"])
        stat_df.to_csv(save_path / "return_stats.csv", index=False)
        print("[INFO] 平均戻りステップ数と回帰率を保存しました。")


def plot_bla_by_cluster_and_year(
    data_df: pd.DataFrame,
    item_cluster_labels: dict,
    save_dir: str,
    min_cluster_samples: int = 50,
    min_sample_per_point: int = 10,
    decay: float = 0.5,
):
    # --- (0) 公開年フィルタ ---
    def extract_year(title: str):
        m = re.search(r"\((\d{4})\)$", title)
        return int(m.group(1)) if m else None

    df = data_df.copy()
    df["release_year"] = df["title"].apply(extract_year)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["timestamp"].dt.year
    df = df.dropna(subset=["release_year"])
    df = df[df["release_year"].astype(int) == df["year"]]

    # --- (1) クラスタ & Exposure 計算 ---
    df["cluster"] = df["title"].map(item_cluster_labels)
    df = df.dropna(subset=["cluster"])
    df = df.sort_values(["userId", "cluster", "timestamp"])
    df["cluster_exposure"] = df.groupby(["userId", "cluster"]).cumcount() + 1

    # --- (2) total_reps 計算 & フィルタ ---
    reps = df.groupby(["userId", "cluster"]).size().reset_index(name="total_reps")
    df = df.merge(reps, on=["userId", "cluster"])
    df = df[(df["total_reps"] >= 5) & (df["total_reps"] <= 30)]

    # --- (3) 小規模クラスタ除外 ---
    valid_clusters = df["cluster"].value_counts()[lambda s: s > 10].index
    df = df[df["cluster"].isin(valid_clusters)]

    # --- (4) BLA 計算（日単位）---
    def compute_bla_series(times: pd.Series):
        out = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, t in enumerate(times):
                if i == 0:
                    out.append(0.0)
                else:
                    past = times.iloc[:i]
                    diffs = (t - past).dt.total_seconds() / 86400.0
                    out.append(np.log((diffs ** (-decay)).sum()))
        return pd.Series(out, index=times.index)

    df["bla"] = (
        df.groupby(["userId", "cluster"])["timestamp"]
        .apply(compute_bla_series)
        .reset_index(level=[0, 1], drop=True)
    )

    # --- (5) inf/NaN を完全に除去 ---
    df = df[np.isfinite(df["bla"])]

    # --- (6) rep_class (累積ユーザー数4分割) ---
    pairs = (
        df[["userId", "cluster", "total_reps"]]
        .drop_duplicates(subset=["userId", "cluster"])
        .sort_values("total_reps")
    )
    total_users = pairs["userId"].nunique()
    uc = (
        pairs.groupby("total_reps")["userId"]
        .nunique()
        .reset_index(name="n_users")
        .sort_values("total_reps")
    )
    uc["cum"] = uc["n_users"].cumsum()
    q25, q50, q75 = total_users * 0.25, total_users * 0.50, total_users * 0.75
    t1 = int(uc.loc[uc["cum"] >= q25, "total_reps"].iloc[0])
    t2 = int(uc.loc[uc["cum"] >= q50, "total_reps"].iloc[0])
    t3 = int(uc.loc[uc["cum"] >= q75, "total_reps"].iloc[0])

    bins = [0, t1 + 1, t2 + 1, t3 + 1, pairs["total_reps"].max() + 1]
    labels = ["LowRep", "ModRep", "HighRep", "VHRep"]
    df["rep_class"] = pd.cut(df["total_reps"], bins=bins, labels=labels, right=False)

    print("\n[INFO] Repetition-class summary:")
    print(
        df.groupby("rep_class")
        .agg(num_users=("userId", "nunique"), num_clusters=("cluster", "nunique"))
        .reset_index()
        .to_string(index=False)
    )

    # --- 出力パス準備 ---
    base = Path(save_dir) / "plt" / "mee_plot"
    by_cluster_dir = base / "by_cluster"
    by_year_dir = base / "by_year"
    by_cluster_dir.mkdir(parents=True, exist_ok=True)
    by_year_dir.mkdir(parents=True, exist_ok=True)

    # --- (7) クラスタ別プロット ---
    cl_agg = (
        df.groupby(["cluster", "rep_class", "cluster_exposure"])
        .agg(
            mean_bla=("bla", "mean"),
            count=("bla", "count"),
            sem_bla=("bla", lambda x: sem(x) if x.count() > 1 else np.nan),
        )
        .reset_index()
    )
    # プロット前にサンプル数不足を除外
    cl_agg = cl_agg[cl_agg["count"] >= min_sample_per_point]

    for cid in sorted(valid_clusters):
        sub = cl_agg[cl_agg["cluster"] == cid]
        if sub.empty:
            continue
        plt.figure(figsize=(8, 5))
        for cls in labels:
            tmp = sub[sub["rep_class"] == cls]
            if tmp.empty:
                continue
            x, y = tmp["cluster_exposure"], tmp["mean_bla"]
            e = 1.96 * tmp["sem_bla"]
            plt.plot(x, y, marker="o", label=cls)
            plt.fill_between(x, y - e, y + e, alpha=0.3)
        plt.title(f"Cluster {cid}: BLA vs Exposure (days)")
        plt.xlabel("Exposure Count")
        plt.ylabel("BLA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(by_cluster_dir / f"cluster_{cid}_bla.png")
        plt.close()

    # --- (8) 年代別サブプロット ---
    yr_agg = (
        df.groupby(["year", "rep_class", "cluster_exposure"])
        .agg(
            mean_bla=("bla", "mean"),
            count=("bla", "count"),
            sem_bla=("bla", lambda x: sem(x) if x.count() > 1 else np.nan),
        )
        .reset_index()
    )
    yr_agg = yr_agg[yr_agg["count"] >= min_sample_per_point]

    years = sorted(yr_agg["year"].unique())
    ncols, nrows = 2, (len(years) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, yr in enumerate(years):
        ax = axes[i]
        sub = yr_agg[yr_agg["year"] == yr]
        for cls in labels:
            tmp = sub[sub["rep_class"] == cls]
            if tmp.empty:
                continue
            x, y = tmp["cluster_exposure"], tmp["mean_bla"]
            e = 1.96 * tmp["sem_bla"]
            ax.plot(x, y, label=cls)
            ax.fill_between(x, y - e, y + e, alpha=0.2)
        ax.set_title(f"Year {yr}")
        ax.set_xlim(0, 30)
        ax.grid(True)
        if i == 0:
            ax.legend()

    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("Exposure Count")
    fig.supylabel("BLA")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(by_year_dir / "bla_by_year.png")
    plt.close()

    print("[INFO] Cluster- and Year-based BLA plots saved.")


def plot_rating_by_year_subplots(
    data_df: pd.DataFrame,
    item_cluster_labels: dict,
    save_dir: str,
    min_cluster_samples: int = 50,
    min_sample_per_point: int = 10,
    masterpiece_threshold: float = 4.3,  # ★追加① 名作とみなす平均評価
    masterpiece_min_votes: int = 50,  # ★追加② 名作判定に必要な最小票数
):
    """
    クラスタ別・反復レベル別に MEE (逆 U 字) を可視化し、
    repetition-class を「(userId, cluster) ごとの累積ユーザー数」
    がほぼ均等になるように４分割して付与。
    * items ≤ 10 本しかないクラスタは除外
    * 作品タイトル末尾の (YYYY) から公開年を抜き出し、
      視聴年と一致する作品のみ対象
    """

    # --- 0. release_year をタイトルから抽出 ---
    def extract_year(title: str):
        m = re.search(r"\((\d{4})\)$", title)
        return int(m.group(1)) if m else None

    # ---------- ★追加③ 名作ラベル付与＆除外 ----------
    item_stats = (
        data_df.groupby("title")["rating"].agg(mean_rating="mean", vote_cnt="count").reset_index()
    )
    masterpiece_titles = item_stats.loc[
        (item_stats["vote_cnt"] >= masterpiece_min_votes)
        & (item_stats["mean_rating"] >= masterpiece_threshold),
        "title",
    ]
    print(f"[INFO] Excluding {len(masterpiece_titles)} masterpieces from analysis")
    data_df = data_df[~data_df["title"].isin(masterpiece_titles)]

    df = data_df.copy()
    df["release_year"] = df["title"].apply(extract_year)
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["year"] = df["timestamp"].dt.year
    # --- 1. rating をユーザ基準化 (z-score) ---
    # grand_mean = df["rating"].mean()
    user_means = df.groupby("userId")["rating"].mean().rename("user_mean")
    # item_means = df.groupby("title")["rating"].mean().rename("item_mean")
    df = df.merge(user_means, on="userId", how="left")
    # df = df.merge(item_means, on="title", how="left")
    df["rating_z"] = df["rating"] - df["user_mean"]  # - df["item_mean"] + grand_mean
    # --- 3. クラスタ付与 & Exposure 計算 ---
    df["cluster"] = df["title"].map(item_cluster_labels)
    df = df.dropna(subset=["cluster"])
    df["rating_z"] = df.groupby("cluster")["rating_z"].transform(
        lambda vals: rft_ratings(vals.tolist(), w=0.5)
    )
    df = df.sort_values(["userId", "timestamp"])
    df["cluster_exposure"] = df.groupby(["userId", "cluster"]).cumcount() + 1

    reps = df.groupby(["userId", "cluster"]).size().reset_index(name="total_reps")
    df = df.merge(reps, on=["userId", "cluster"])
    df = df[(df["total_reps"] >= 5) & (df["total_reps"] <= 50)]

    # --- 4. 小規模クラスタ(≤10)を除外 ---
    valid_clusters = df["cluster"].value_counts()[lambda s: s > 10].index
    df = df[df["cluster"].isin(valid_clusters)]

    # --------------------------------------------------

    # --- 6. repetition-class 付与 ---
    bins = [0, 10, 20, 30, 50]
    labels = ["LowRep", "ModRep", "HighRep", "VHRep"]
    df["rep_class"] = pd.cut(df["total_reps"], bins=bins, labels=labels, right=False)

    print("\n[INFO] Counts per repetition-class:")
    for cls in labels:
        sub = df[df["rep_class"] == cls]
        title_count = sub["title"].nunique()
        cluster_count = sub["cluster"].nunique()
        print(f"  {cls}: titles = {title_count}, clusters = {cluster_count}")

    # --- 7. 全クラスタ統合プロット (rep_class別) ---
    agg = (
        df.groupby(["rep_class", "cluster_exposure"])["rating_z"]
        .agg(mean="mean", sem=sem, count="count")
        .reset_index()
    )
    rep_class_styles = {
        "LowRep": ("blue", "o"),
        "ModRep": ("orange", "s"),
        "HighRep": ("green", "X"),
        "VHRep": ("red", "D"),
    }

    plt.figure(figsize=(10, 6))
    for cls in labels:
        tmp = agg[agg["rep_class"] == cls]
        if tmp.empty:
            continue
        x = tmp["cluster_exposure"]
        y = tmp["mean"]
        e = 1.96 * tmp["sem"]
        color, marker = rep_class_styles[cls]
        plt.errorbar(
            x,
            y,
            yerr=e,
            fmt=f"-{marker}",
            color=color,
            capsize=1,
            elinewidth=0.5,
            markeredgewidth=1,
            label=cls,
        )

    plt.title("Average Z-scored Rating vs Cluster Exposure (All Repetition Classes)")
    plt.xlim(0, 30)
    plt.xlabel("Exposure Count to Same Cluster")
    plt.ylabel("Average Z-scored Rating")
    plt.grid(True)
    plt.legend()
    base_path = Path(save_dir) / "plt" / "mee_plot"
    base_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(base_path / "all_rep_classes_mee_plot.png")
    plt.close()

    # --- 8. repetition-class ごとの 2 次回帰結果表示 (x ≤ 30) ---
    print("\n[INFO] Quadratic fit per repetition-class (x ≤ 30)")
    for cls in labels:
        df_sub = df[(df["rep_class"] == cls) & (df["cluster_exposure"] <= 30)]
        if df_sub.empty:
            continue
        tmp = df_sub.groupby("cluster_exposure")["rating_z"].mean().reset_index()
        x, y = tmp["cluster_exposure"].values, tmp["rating_z"].values
        X = np.column_stack([np.ones_like(x), x, x**2])
        model = sm.OLS(y, X).fit()
        print(
            f"  {cls:7s}: R²={model.rsquared:.3f} , β₂={model.params[2]:+.4f} (p={model.pvalues[2]:.3g})"
        )

        # --- 8. ブートストラップ検証関数 ---

    def bootstrap_beta2_for_class(df_sub, B=1000, seed=42):
        rng = np.random.RandomState(seed)
        beta2_vals = []
        for _ in range(B):
            boot_df = resample(df_sub, replace=True, n_samples=len(df_sub), random_state=rng)
            x_b, y_b = boot_df["cluster_exposure"].values, boot_df["rating_z"].values
            X_b = np.column_stack([np.ones_like(x_b), x_b, x_b**2])
            m = sm.OLS(y_b, X_b).fit()
            beta2_vals.append(m.params[2])
        arr = np.array(beta2_vals)
        ci_low, ci_up = np.percentile(arr, [2.5, 97.5])
        p_boot = 2 * np.mean(arr > 0)
        return ci_low, ci_up, p_boot

    # --- 9. ブートストラップによる β₂ の頑健性検証 ---
    print("\n[INFO] Bootstrap validation of β₂ for each repetition class")
    for cls in labels:
        df_sub = df[(df["rep_class"] == cls) & (df["cluster_exposure"] <= 30)]
        if len(df_sub) < min_sample_per_point:
            continue
        ci_low, ci_up, p_boot = bootstrap_beta2_for_class(df_sub, B=1000)
        print(f"  {cls:7s}: β₂ 95% CI [{ci_low:.5f}, {ci_up:.5f}], bootstrap p≈{p_boot:.3f}")
    # --- 9. クラスタ別プロット (errorbarで信頼区間を表示) ---
    cl_agg = (
        df.groupby(["cluster", "rep_class", "cluster_exposure"])["rating_z"]
        .agg(mean="mean", sem=sem, count="count")
        .reset_index()
    )
    cluster_dir = base_path / "by_cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for cid in sorted(valid_clusters):
        sub = cl_agg[cl_agg["cluster"] == cid]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 6))
        for cls in labels:
            tmp = sub[(sub["rep_class"] == cls) & (sub["count"] >= min_sample_per_point)]
            if tmp.empty:
                continue
            x = tmp["cluster_exposure"]
            y = tmp["mean"]
            e = 1.96 * tmp["sem"]
            color, marker = rep_class_styles[cls]
            plt.errorbar(
                x,
                y,
                yerr=e,
                fmt=f"-{marker}",
                color=color,
                capsize=3,
                elinewidth=1,
                markeredgewidth=1,
                label=cls,
            )
        plt.title(f"Cluster {cid}: Z-scored Rating vs Exposure (n≥{min_cluster_samples})")
        plt.xlabel("Exposure Count to Same Cluster")
        plt.ylabel("Average Z-scored Rating")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(cluster_dir / f"cluster_{cid}_mee_plot.png")
        plt.close()

    # --- 10. クラスタ固定効果付き2次回帰 ---
    print("\n[INFO] クラスタ固定効果付き2次回帰（x ≤ 30）")
    df_sub = df[df["cluster_exposure"] <= 30].copy()
    if df_sub.empty:
        print("データがありません。")
    else:
        df_sub["exposure_sq"] = df_sub["cluster_exposure"] ** 2
        formula = "rating_z ~ cluster_exposure + exposure_sq + C(cluster)"
        model = sm.OLS.from_formula(formula, data=df_sub).fit()
        print(model.summary())
        print(
            "\n[INFO] exposure_sq の係数: {:.4f}, p-value: {:.4g}".format(
                model.params["exposure_sq"], model.pvalues["exposure_sq"]
            )
        )

    return df


def rft_ratings(values, w=0.5):
    """
    Parducci の RFT（範囲–周波数理論）を NumPy で高速実装。
    values: list[float] 文脈中の評価値リスト
    w: 範囲原理の重み (0≤w≤1)
    """
    arr = np.asarray(values, dtype=float)  # リスト→NumPy 配列化
    mn, mx = (
        arr.min(),
        arr.max(),
    )  # 最小・最大の取得は O(n)
    # 範囲原理：配列化された演算で一括正規化
    if mx > mn:
        R = (arr - mn) / (mx - mn)
    else:
        R = np.full_like(arr, 0.5)

    # 頻度原理：np.sort＋searchsorted で O(n log n) → O(n) 近似に
    sorted_vals = np.sort(
        arr
    )  # 内部では速いソート実装  [oai_citation:1‡llego.dev](https://llego.dev/posts/numpy-sorting-arrays/?utm_source=chatgpt.com)
    # each element’s rank = 挿入位置 / (N-1)
    N = arr.size
    if N > 1:
        F = np.searchsorted(sorted_vals, arr, side="right") / (
            N - 1
        )  # O(n log n)  [oai_citation:2‡NumPy](https://numpy.org/doc/2.2/reference/generated/numpy.searchsorted.html?utm_source=chatgpt.com)
    else:
        F = np.full_like(arr, 0.5)

    # 加重平均もベクトルで一括
    J = w * R + (1 - w) * F
    return J.tolist()


def detect_non_invU_users(df: pd.DataFrame, x_range=(30, 60), min_exposures: int = 30):
    """
    各ユーザー・クラスタごとに二次回帰をあてはめ、
    二次項係数が正（convex: 上に凸）になっているペアを抽出する。

    Parameters:
      df: DataFrame
        必須列: ['userId', 'title', 'cluster_exposure', 'rating_z', 'cluster']
      min_exposures, max_exposures: int
        exposure 回数の範囲フィルタ（データ点の数）
      x_range: tuple
        x（cluster_exposure）の範囲フィルタ（例: (30, 50)）
    Returns:
      List[(userId, clusterId)]
    """

    convex_pairs = []
    x_min, x_max = x_range
    for (user, cluster), grp in df.groupby(["userId", "cluster"]):
        # xを範囲指定でフィルタ
        mask = (grp["cluster_exposure"] >= x_min) & (grp["cluster_exposure"] <= x_max)
        sub_grp = grp.loc[mask]

        x = sub_grp["cluster_exposure"].values
        y = sub_grp["rating_z"].values

        # exposure 回数の範囲チェック
        if min_exposures <= len(x):
            # 2次多項式フィット
            coefs = np.polyfit(x, y, 2)
            # 二次項が正ならconvex（上に凸）
            if coefs[0] > 0:
                convex_pairs.append((user, cluster))
        else:
            print("No User not inversed shaped over 30 expousres")
    return convex_pairs


def build_cluster_transition_network_for_group(
    data_df,
    item_cluster_labels: dict,
    save_dir: Path,
    min_count: int = 1,
    prob_threshold: float = 0.01,
    highlight_top_k: int = 50,
):
    """
    ユーザー群のクラスタ間遷移ネットワークを“確率”ベースで可視化します。
    ・min_count       : 遷移回数がこの値以上のエッジのみ対象
    ・prob_threshold  : 確率がこの値以上のエッジのみ描画
    ・highlight_top_k : 確率上位 K 本のエッジを太く／濃く強調
    """

    # 1) 遷移カウント集計
    transition_counts = defaultdict(int)
    for _, grp in data_df.groupby("userId"):
        seq = [
            item_cluster_labels.get(t)
            for t in grp.sort_values("timestamp")["title"]
            if t in item_cluster_labels
        ]
        for src, dst in zip(seq, seq[1:]):
            if src is not None and dst is not None and src != dst:
                transition_counts[(src, dst)] += 1

    # 2) グラフ構築＋min_count フィルタ
    G = nx.DiGraph()
    for (src, dst), cnt in transition_counts.items():
        if cnt >= min_count:
            G.add_edge(src, dst, count=cnt)
    if G.number_of_edges() == 0:
        print("遷移データがありません。閾値を緩めてください。")
        return

    # 3) 各ノードからの出力合計を計算し、prob 属性を追加
    out_sum = {u: sum(d["count"] for _, _, d in G.out_edges(u, data=True)) for u in G.nodes()}
    for u, v, d in G.edges(data=True):
        total = out_sum.get(u, 0)
        d["prob"] = (d["count"] / total) if total > 0 else 0.0

    # 4) レイアウト
    pos = nx.kamada_kawai_layout(G, weight=None)

    # 5) エッジ情報収集＆上位 K 強調セット
    edges = list(G.edges(data=True))
    sorted_edges = sorted(edges, key=lambda x: x[2]["prob"], reverse=True)
    top_edges = {(u, v) for u, v, _ in sorted_edges[:highlight_top_k]}

    # 6) カラーマッピング（クラスタ数分の落ち着いた色）
    unique_cids = sorted(G.nodes())
    n = len(unique_cids)
    pal = sns.color_palette("husl", n)
    cluster_to_color = {cid: pal[i] for i, cid in enumerate(unique_cids)}

    # 7) ノードサイズ（出力中心性）
    outdeg = nx.in_degree_centrality(G)
    node_sizes = [100 + 3000 * outdeg.get(cid, 0) for cid in unique_cids]

    # 8) 描画
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=unique_cids,
        node_size=node_sizes,
        node_color=[cluster_to_color[cid] for cid in unique_cids],
        edgecolors="white",
        alpha=0.9,
    )
    for u, v, d in edges:
        p = d["prob"]
        if p < prob_threshold:
            continue
        if (u, v) in top_edges:
            width, alpha = 1.0 + 4.0 * p, min(p * 1.5, 1)
        else:
            width, alpha = 1.0 + 2.0 * p, p
        arrow_size = 10 + 10 * p
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color="gray",
            alpha=alpha,
            arrowstyle="<->",
            arrows=True,
            arrowsize=arrow_size,
        )

    plt.title(f"Cluster Transition Network (n_users={data_df['userId'].nunique()})")
    plt.axis("off")

    # 9) 昇順ソートした凡例
    legend_handles = [
        Patch(facecolor=cluster_to_color[cid], edgecolor="black", label=f"Cluster {cid}")
        for cid in unique_cids
    ]
    plt.legend(
        handles=legend_handles,
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    plt.subplots_adjust(right=0.75)

    # 10) 保存
    out_dir = save_dir / "plt" / "network" / "cluster_group"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cluster_group_transition_network.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[INFO] ネットワークを保存しました: {out_path}")


def main():
    return


if __name__ == "__main__":
    main()
