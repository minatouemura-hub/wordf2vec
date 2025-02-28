import csv
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from word2vec.train import Trainer


class Project_On(Trainer):
    def __init__(
        self,
        base_dir: Path,
        folder_path: Path,
        weight_path: Path,
        top_k=10,
        embedding_dim=50,
        num_negatives=5,
        batch_size=32,
        epochs=5,
        learning_rate=0.005,
    ):
        super().__init__(
            folder_path,
            weight_path,
            embedding_dim,
            num_negatives,
            batch_size,
            epochs,
            learning_rate,
        )
        self.base_dir = base_dir
        self.top_k = top_k
        if os.path.isfile(weight_path):
            self._read_weight_vec(device="cpu")
            self.book_embeddings = self.book_embed.weight.data.cpu().numpy()
        else:
            self.run_train()

    def projection(self):
        print("\n==射影を開始します==")
        social_axis = self.find_gender_axis(self.top_k)
        normed_social_axis = social_axis / np.linalg.norm(social_axis)
        # 進捗管理したいので計算速度落ちてもって感じ...
        projections = np.empty((self.book_embeddings.shape[0],))
        for i in tqdm(range(self.book_embeddings.shape[0]), desc="Projecting embeddings"):
            # for i in tqdm(range(1, 100, 1), desc="Projecting embeddings"):
            projections[i] = np.dot(self.book_embeddings[i], normed_social_axis)
        self._save_projections(projections)

    def _save_projections(self, projections):
        RESULT_PATH = self.base_dir / "result" / "projection_result.csv"
        with open(RESULT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["book_title", "projection_vec"])
            for idx, proj in enumerate(projections):
                book_title = self.id2book.get(idx, "Unknow")
                writer.writerow([book_title, proj])
        print(f"==射影の結果は{RESULT_PATH}に保存されています==")
