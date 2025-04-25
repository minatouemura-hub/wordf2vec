import argparse
from dataclasses import dataclass, field
from typing import Any, Dict


# == DataClass ==
@dataclass
class WholeConfig:
    dataset: str = "Movie"
    retrain: bool = True
    grid_search_flag: bool = True
    fast_greedy_compare: bool = True


@dataclass
class TrainerConfig:
    t_range: int = 100


@dataclass
class Word2VecCongig:
    # == grid_search用 ==
    # sample_range: list = field(default_factory=lambda: [1e-4, 1e-5])
    negative_range: list = field(default_factory=lambda: [35, 50, 65, 80])  # negative sampling
    alpha_range: list = field(default_factory=lambda: [0.01, 0.05])  # 初期学習率
    size_range: list = field(default_factory=lambda: [100, 150, 200])  # 埋め込みベクトルの次元
    n_trials: int = 20
    # ==
    # down_sampleは基本on
    down_sample: bool = True
    sample: float = 1e-4

    # 1回分の学習用
    embedding_dim: int = 150
    num_negatives: int = 35
    batch_size: int = 124
    epochs: int = 10
    learning_rate: float = 0.01
    scheduler_factor: float = 0.5
    early_stop_threshold: float = 0.01
    min_user_cnt: int = (
        1  # 1000ユーザーなら2000冊くらいからanalogy_taskを解けるようになる．13万冊を扱うなら...7万人くらいのデータ
    )


@dataclass
class ClusterConfig:
    tolerance_ratio: float = 0.15
    max_iter: int = 200
    cluster_search_range: list = field(default_factory=lambda: [10, 50])
    tol: float = 1e-4
    random_state: int = 42


@dataclass
class NetworkConfig:
    item_weight: int = 50
    cluster_user_weight: int = 70
    base_user_weight: int = 10


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="word2vecf")

    parser.add_argument(
        "--dataset", type=str, default="Movie10M", choices=["Book", "Movie1M", "Movie10M"]
    )
    parser.add_argument("--retrain", action="store_true", help="Retraining word2vec flag")
    parser.add_argument(
        "--recompute_axis", action="store_true", help="Recompute social dimention flag"
    )
    parser.add_argument(
        "--grid_search_flag", action="store_true", help="Flag for whether grid_search"
    )
    parser.add_argument(
        "--fast_greedy_compare",
        action="store_true",
        help="Flag for comparison between ours model vs FG",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="sim_task",
        choices=["analogy_task", "sim_task"],
        help="Task to eval accuracy of Word2vec model ",
    )

    args = parser.parse_args()
    return vars(args)


# dict => dataclass
def parse_config(args_dict: Dict[str, Any]):
    whole_keys = WholeConfig.__annotations__.keys()

    whole_config = WholeConfig(**{k: args_dict[k] for k in whole_keys})
    word2vegconfig = Word2VecCongig()
    networkcofig = NetworkConfig()
    return (whole_config, word2vegconfig, networkcofig)
