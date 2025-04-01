import argparse
from dataclasses import dataclass, field
from typing import Any, Dict


# == DataClass ==
@dataclass
class WholeConfig:
    retrain: bool = True
    recompute_axis: bool = True  # 社会軸の再計算
    grid_search_flag: bool = True


@dataclass
class Word2VecCongig:
    # sample_range: list = [1e-3, 1e-4]  # 高頻度のダウンサンプリング
    negative_range: list = field(default_factory=lambda: [10, 35])  # negative sampling
    alpha_range: list = field(default_factory=lambda: [0.01, 0.1])  # 初期学習率
    size_range: list = field(default_factory=lambda: [100, 150])  # 埋め込みベクトルの次元
    # down_sampleの追加
    down_sample: bool = True
    sample: float = 1e-4
    embedding_dim: int = 150
    num_negatives: int = 15
    batch_size: int = 124
    epochs: int = 20
    learning_rate: float = 0.01


@dataclass
class SocialAxisConfig:
    how_dim_reduce: str = "mean"  # 複数のベクトルをどのように1つの社会軸に落とすか
    find_axis_pairs: int = 5  # 何個のpairから社会軸を作成するか


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="word2vecf")

    parser.add_argument("--retrain", action="store_true", help="Retraining word2vec flag")
    parser.add_argument(
        "--recompute_axis", action="store_true", help="Recompute social dimention flag"
    )
    parser.add_argument("--grid_search_flag", action="store_true")

    parser.add_argument("--how_dim_reduce", type=str, default="pca", choices=["pca", "mean"])
    parser.add_argument("--find_axis_pairs", type=int, default=5)

    args = parser.parse_args()
    return vars(args)


# dict => dataclass
def parse_config(args_dict: Dict[str, Any]):
    whole_keys = WholeConfig.__annotations__.keys()
    social_axis_keys = SocialAxisConfig.__annotations__.keys()

    whole_config = WholeConfig(**{k: args_dict[k] for k in whole_keys})
    social_axis_config = SocialAxisConfig(**{k: args_dict[k] for k in social_axis_keys})
    word2vegconfig = Word2VecCongig()
    return whole_config, social_axis_config, word2vegconfig
