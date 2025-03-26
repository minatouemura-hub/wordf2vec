import os  # noqa
from pathlib import Path

import pandas as pd  # noqa

from cluster_analysis.analysis import ClusterAnalysis
from gender_axis.projection import Project_On  # noqa
from word2vec.train import Trainer  # noqa


def main():
    BASE_DIR = Path(__file__).resolve().parent  # noqa
    DATA_PATH = BASE_DIR / "data" / "all_users_results.json"
    WEIGHT_PATH = BASE_DIR / "weight_vec" / "book2vec_model.pth"
    PROJECTION_RESULT = BASE_DIR / "result" / "projection_result.csv"  # noqa

    clster_analysis = ClusterAnalysis(
        base_dir=BASE_DIR,
        folder_path=DATA_PATH,
        weight_path=WEIGHT_PATH,
        result_path=PROJECTION_RESULT,
        num_cluster=20,
        num_samples=20000,
    )

    clster_analysis.k_means_tsne_plt()
    clster_analysis.cluster_distribution()
    # clster_analysis.make_correlogram_from_dict()
    clster_analysis.dtw_kmeans_users(max_clusters=5)


if __name__ == "__main__":
    main()
