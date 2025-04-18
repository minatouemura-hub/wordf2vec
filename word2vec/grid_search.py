import itertools
from pathlib import Path

import optuna
from optuna.trial import Trial
from tqdm import tqdm

from arg import Word2VecCongig

from .preprocessor import Movie1MDataset
from .trainer import Trainer


class GridSearch:
    def __init__(self, grid_config: Word2VecCongig, weight_path, dataset: "Movie1MDataset"):
        self.grid_config = grid_config
        self.weight_path = weight_path
        self.dataset = dataset
        self.results = []

    def grid_search(self):
        best_score = -1
        best_params = {}

        # 全組み合わせをリストに展開（tqdmで進捗を追いやすくする）
        param_combinations = list(
            itertools.product(
                self.grid_config.negative_range,
                self.grid_config.alpha_range,
                self.grid_config.size_range,
            )
        )

        print(f"=== Grid Search 開始: {len(param_combinations)} 試行 ===")

        for negative, alpha, size in tqdm(param_combinations, desc="Grid Search"):
            # 各試行のログ
            print(f"\n→ 試行: negative={negative}, alpha={alpha}, size={size}")

            # Trainerに渡すパラメータ構成
            trainer = Trainer(
                weight_path=self.weight_path,
                dataset=self.dataset,
                embedding_dim=size,
                num_negatives=negative,
                learning_rate=alpha,
                batch_size=self.grid_config.batch_size,
                epochs=self.grid_config.epochs,
                scheduler_factor=self.grid_config.scheduler_factor,
                early_stop_threshold=self.grid_config.early_stop_threshold,
                grid_search=True,
            )

            # モデル学習 + 評価（trainer.trainがaccを返す想定）
            acc = trainer.train()

            result = {
                "negative": negative,
                "alpha": alpha,
                "size": size,
                "acc": acc,
            }
            self.results.append(result)

            # ベストスコア更新
            if acc > best_score:
                best_score = acc
                best_params = result

        print("\n=== Grid Search 完了 ===")
        print(f"ベストスコア: {best_score:.4f}")
        print(f"ベストパラメータ: {best_params}")
        return best_params, best_score


class OptunaSearch:
    def __init__(self, dataset, weight_path: Path, word2vec_config: Word2VecCongig):
        self.dataset = dataset
        self.weight_path = weight_path
        self.word2vec_config = word2vec_config
        self.n_trials = word2vec_config.n_trials

    def _objective(self, trial: Trial):
        # Optunaによりサンプリングされたハイパーパラメータ
        embedding_dim = trial.suggest_categorical("embedding_dim", self.word2vec_config.size_range)
        num_negatives = trial.suggest_categorical(
            "num_negatives", self.word2vec_config.negative_range
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            self.word2vec_config.alpha_range[0],
            self.word2vec_config.alpha_range[1],
            log=True,
        )
        # sample = trial.suggest_categorical("sample", [1e-3, 1e-4, 1e-5])

        # Trainer インスタンスを構築して学習
        trainer = Trainer(
            weight_path=self.weight_path,
            dataset=self.dataset,
            embedding_dim=embedding_dim,
            num_negatives=num_negatives,
            learning_rate=learning_rate,
            batch_size=self.word2vec_config.batch_size,
            epochs=self.word2vec_config.epochs,
            scheduler_factor=self.word2vec_config.scheduler_factor,
            early_stop_threshold=self.word2vec_config.early_stop_threshold,
            grid_search=True,
        )

        acc = trainer.train()  # アナロジータスク精度を返すと仮定
        return acc  # maximize

    def search(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials)

        print("Best score:", study.best_value)
        print("Best params:", study.best_params)
        return study.best_params, study.best_value
