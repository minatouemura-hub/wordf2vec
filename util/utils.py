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
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from umap import UMAP  # noqa


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
    n_neighbors_list = [30, 40, 50, 60]
    min_dist_list = [0.01, 0.01, 0.1]

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
                spread=1.0,
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
    print(f"Best param:{n_nb},{m_dist}")
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
    # df["rating_z"] = df.groupby("cluster")["rating_z"].transform(
    #     lambda vals: rft_ratings(vals.tolist(), w=0.5)
    # )
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
        user_count = sub["userId"].nunique()  # 追加：ユニークユーザー数
        print(
            f"  {cls}: titles = {title_count}, "
            f"clusters = {cluster_count}, "
            f"users = {user_count}"
        )

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

    # # --- 10. クラスタ固定効果付き2次回帰 ---
    # print("\n[INFO] クラスタ固定効果付き2次回帰（x ≤ 30）")
    # df_sub = df[df["cluster_exposure"] <= 30].copy()
    # if df_sub.empty:
    #     print("データがありません。")
    # else:
    #     df_sub["exposure_sq"] = df_sub["cluster_exposure"] ** 2
    #     formula = "rating_z ~ cluster_exposure + exposure_sq + C(cluster)"
    #     model = sm.OLS.from_formula(formula, data=df_sub).fit()
    #     print(model.summary())
    #     print(
    #         "\n[INFO] exposure_sq の係数: {:.4f}, p-value: {:.4g}".format(
    #             model.params["exposure_sq"], model.pvalues["exposure_sq"]
    #         )
    #     )

    return df


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
