import os
from pathlib import Path

from word2vec.train import Trainer


def main():
    BASE_DIR = Path(__file__).resolve().parent  # noqa
    DATA_PATH = BASE_DIR / "data" / "all_users_results.json"
    WEIGHT_PATH = BASE_DIR / "weight_vec" / "book2vec_model.pth"
    trainer = Trainer(folder_path=DATA_PATH, weight_path=WEIGHT_PATH)
    if os.path.isfile(WEIGHT_PATH):
        trainer.read_weight_vec(WEIGHT_PATH)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
