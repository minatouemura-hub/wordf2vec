import os  # noqa
from pathlib import Path

from gender_axis.projection import Project_On
from word2vec.train import Trainer  # noqa


def main():
    BASE_DIR = Path(__file__).resolve().parent  # noqa
    DATA_PATH = BASE_DIR / "data" / "all_users_results.json"
    WEIGHT_PATH = BASE_DIR / "weight_vec" / "book2vec_model.pth"
    projecter = Project_On(
        base_dir=BASE_DIR, folder_path=DATA_PATH, weight_path=WEIGHT_PATH, top_k=10
    )
    projecter.projection()


if __name__ == "__main__":
    main()
