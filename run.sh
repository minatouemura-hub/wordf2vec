#!/bin/bash
#SBATCH -p gpu_short                # ← 短期GPUバッチ用
#SBATCH --gres=gpu:2               # ← GPUを2枚要求
#SBATCH -c 8                       # ← CPUも4スレッド使用
#SBATCH --time=4:00:00             # ← 最大4時間までOK
#SBATCH -J book_vec_train          # ← ジョブ名
#SBATCH -o logs/%x_%j.out          # ← 標準出力ログ
#SBATCH -e logs/%x_%j.err          # ← 標準エラー出力ログ

#SBATCH --mail-user=uemura.minato.uk7@naist.ac.jp   # ← 完了通知先メールアドレス
#SBATCH --mail-type=END,FAIL                            # ← ジョブ終了時にメール送信

source /work/minato-u/miniconda3/etc/profile.d/conda.sh  # conda 初期化
conda activate network_env                                    # 仮想環境をアクティベート
python main.py    