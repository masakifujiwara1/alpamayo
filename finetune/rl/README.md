# Cosmos RL を使った Alpamayo RL Post-training

このディレクトリには、Cosmos-RL 経由の GRPO を使って Alpamayo モデル (離散 action-token variant を持つ VLM backbone) をファインチューニングするための RL post-training コードが含まれています。このコードは [Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) と [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) の両方をサポートします。

<p align="center">
  <img src="assets/alpamayo_rl_framework.png" alt="Alpamayo RL Framework" width="700">
</p>

## 目次

1. [はじめに](#getting-started)
   1. [Python 環境](#1-python-environment)
   2. [環境変数](#2-environment-variables)
   3. [HuggingFace 認証](#3-authenticate-with-huggingface)
   4. [Alpamayo モデルのダウンロードと準備](#4-download-and-prepare-the-alpamayo-model)
   5. [Physical AI データセットのサブセットをダウンロード](#5-download-a-subset-of-the-physical-ai-dataset)
   6. [RL 学習の起動](#6-launch-rl-training)
   7. [推論用に RL チェックポイントを export](#7-export-the-rl-checkpoint-for-inference)
2. [パイプライン概要](#pipeline-overview)
   1. [アーキテクチャ](#architecture)
   2. [主要ファイル](#key-files)
   3. [主要パラメータ](#key-parameters)
   4. [報酬](#reward)
3. [マルチノード大規模学習](#multi-node-large-scale-training)
4. [FAQ](#faq)

<a id="getting-started"></a>
## はじめに

このセクションでは、環境構築から小さなデータセットでの短い RL 学習ジョブの起動まで、**単一ノードでのローカル検証実行**を順に説明します。目的は、マルチノードクラスタ学習へスケールする前に、パイプライン全体 (モデル読み込み、rollout 生成、報酬計算、GRPO 学習) が end-to-end で動作することを確認することです。

> **ハードウェア要件:** ローカルテスト config には、少なくとも **5 GPU** が必要で、各 GPU は 80 GB 以上の VRAM を備えている必要があります。

まず作業ディレクトリを定義します (以降のコマンドはすべて `$YOUR_HOME` を参照します)。

```bash
export YOUR_HOME="/path/to/your/workspace"
```

<a id="1-python-environment"></a>
### 1. Python 環境

```bash
export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"

cd "$YOUR_HOME/alpamayo"
uv venv alpamayo_env
source alpamayo_env/bin/activate
uv sync --active --no-install-package flash-attn   # flash-attn 以外の依存関係をすべてインストール
uv sync --active                                   # その後 flash-attn をビルド (torch が必要)
```

<a id="2-environment-variables"></a>
### 2. 環境変数

セッションごとに次を設定します (または `~/.bashrc` に追加します)。

```bash
# ── Paths ────────────────────────────────────────────────────────
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo"
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_cosmos_rl_job/logs"

# ── Cache ────────────────────────────────────────────────────────
export HF_HOME="$YOUR_HOME/.cache/huggingface"

# ── Runtime ──────────────────────────────────────────────────────
export WANDB_API_KEY="<your_wandb_api_key>"
```

> **Tip:** HuggingFace Hub の rate limit に当たる場合は、`export HF_HUB_OFFLINE=1` と `export TRANSFORMERS_OFFLINE=1` を設定し、モデル/トークナイザの読み込みをすべてローカルキャッシュから行わせてください。

| 変数                     | 必須        | 目的                                                                                           |
| ------------------------ | ----------- | ---------------------------------------------------------------------------------------------- |
| `ALPAMAYO_WORKSPACE`     | yes         | `alpamayo` checkout のルート                                                                   |
| `ALPAMAYO_MODEL_DIR`     | yes         | 事前学習済み Alpamayo モデルディレクトリ (step 4 の出力)                                      |
| `ALPAMAYO_PAI_LOCAL_DIR` | yes         | PAI データセットルート (step 5 の出力)。実行時に entry script から読み込まれます              |
| `ALPAMAYO_LOG_DIR`       | yes         | Cosmos-RL ログ用ディレクトリ                                                                   |
| `UV_CACHE_DIR`           | recommended | uv cache の場所 (step 1 で `uv venv` の前に設定)                                               |
| `HF_HOME`                | recommended | HuggingFace cache の場所                                                                       |
| `HF_HUB_OFFLINE`         | optional    | `1` にすると HuggingFace Hub 呼び出しを省略します (rate limit や air-gapped cluster で有用)    |
| `TRANSFORMERS_OFFLINE`   | optional    | `HF_HUB_OFFLINE` と併せて `1` に設定します                                                     |
| `WANDB_API_KEY`          | recommended | Weights & Biases API key。`[logging].logger = ["console"]` を使う場合は省略できます           |

<a id="3-authenticate-with-huggingface"></a>
### 3. HuggingFace 認証

モデルとデータセットは gated resource へのアクセスを必要とします。以下からアクセスを申請してください。

- [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo 1 Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [Alpamayo 1.5 Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

その後、認証します。

```bash
hf auth login
```

トークンはこちらで取得できます: https://huggingface.co/settings/tokens

<a id="4-download-and-prepare-the-alpamayo-model"></a>
### 4. Alpamayo モデルのダウンロードと準備

HuggingFace のリリースモデルを、学習可能なチェックポイントへ変換します。デフォルトでは、スクリプトは [Alpamayo 1.5 Model Weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B) をダウンロードします。必要に応じて [Alpamayo 1 Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B) に切り替えるには `--alpamayo-model` を使います。

```bash
cd "$ALPAMAYO_WORKSPACE"

python scripts/convert_release_config_to_training.py \
  --output-dir "$ALPAMAYO_MODEL_DIR"
```

<a id="5-download-a-subset-of-the-physical-ai-dataset"></a>
### 5. Physical AI データセットのサブセットをダウンロード

```bash
python scripts/download_pai.py \
  --chunk-ids 3116 \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"
```

次に、ローカル RL 学習用に 16 個の driving clip からなる mini subset を作成します。

```bash
python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk 3116 \
  --num-samples 16 \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index_mini.parquet"
```

<a id="6-launch-rl-training"></a>
### 6. RL 学習の起動

起動前に TOML config を更新してください。ローカルテストには `finetune/rl/toml/alpamayo_rvla_rl_local_test.toml` を使います。

設定すべき主な field:

1. `[train].output_dir`: チェックポイントと学習 artifact の出力先 (例: `$YOUR_HOME/alpamayo_cosmos_rl_job/outputs`)。
2. `[policy].model_name_or_path`: `$ALPAMAYO_MODEL_DIR` に設定します。
3. `[policy.parallelism].dp_shard_size`: ローカルテスト (1 node) では `4`、cluster (multi-node) では `8`。
4. 任意: `[logging].logger = ["console", "wandb"]`。

**ローカルテスト (単一ノード):** インストール済み環境を activate し、alpamayo ディレクトリから実行します。

```bash
cd "$ALPAMAYO_WORKSPACE"
cosmos-rl \
  --config finetune/rl/toml/alpamayo_rvla_rl_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

- `--policy 1 --rollout 1`: policy replica 1 個と rollout replica 1 個を起動します。これは TOML config の `n_init_replicas` を上書きします。

- 学習ログは `$ALPAMAYO_LOG_DIR/logs_<YYYYMMDD-HHMMSS>/` に、プロセスごとに 1 ファイルずつ書き込まれます。

  | ログファイル      | プロセス             | 内容                                                                                             |
  | ----------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
  | `controller.log`  | Cosmos-RL controller | rollout dispatch、step ごとの reward stats、buffer status (`pending rollouts`)、weight sync event |
  | `policy_<i>.log`  | Policy replica *i*   | モデル読み込み、training loss、gradient norm、checkpoint 保存、rank ごとのデータ分布             |
  | `rollout_<i>.log` | Rollout replica *i*  | vLLM engine 起動、generation throughput、weight receive event、sample ごとの reward computation   |

デフォルト設定では、training reward が増加し、trajectory L2 error が低下するはずです (reward は -0.28 から -0.21 に改善し、trajectory L2 は 1.66 から 1.34 に低下)。ローカルテストは、単一の 8 GPU (H100) ノードで約 10 分以内に完了します。

<p align="center">
  <img src="assets/local_training_reward_curves.png" alt="Local training reward curves" width="700">
</p>

<a id="7-export-the-rl-checkpoint-for-inference"></a>
### 7. 推論用に RL チェックポイントを export

Cosmos-RL は、`<output_dir>/checkpoints/step_<N>/policy/` 配下に rank ごとの PyTorch ファイル (`model_rank_<r>.pth`) としてチェックポイントを保存します。これらのファイルには DTensor shard が含まれるため、`ReasoningVLA.from_pretrained()` で直接読み込むことはできません。

policy checkpoint を標準的な HuggingFace checkpoint ディレクトリに変換するには、次を実行します。

```bash
cd "$ALPAMAYO_WORKSPACE"

python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt "$YOUR_HOME/alpamayo_cosmos_rl_job/outputs/checkpoints/step_<N>/policy" \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir "$YOUR_HOME/alpamayo_cosmos_rl_job/exported_model"
```

- `--cosmos-policy-ckpt`: Cosmos-RL policy checkpoint ディレクトリへのパス (`model_rank_*.pth` ファイルを含む)。
- `--base-hf-ckpt`: step 4 で生成した学習用チェックポイントディレクトリ (`$ALPAMAYO_MODEL_DIR`)。Config、tokenizer、processor ファイルはここからコピーされます。
- `--output-dir`: export した HF checkpoint の書き込み先。

export した checkpoint は、推論用に次のように読み込めます。

```python
from reasoning_vla.base_model import RLWrapperReasoningVLA

model = RLWrapperReasoningVLA.from_pretrained(
    "$YOUR_HOME/alpamayo_cosmos_rl_job/exported_model"
)
```

データ読み込み、推論実行、reasoning / trajectory / chain-of-thought の可視化を含む完全な end-to-end 例は、[`notebooks/rl_checkpoint_inference.ipynb`](notebooks/rl_checkpoint_inference.ipynb) を参照してください。

<a id="pipeline-overview"></a>
## パイプライン概要

<a id="architecture"></a>
### アーキテクチャ

Alpamayo RL post-training は、Physical AI workload 向けのスケーラブルな強化学習フレームワークである [Cosmos-RL](https://github.com/NVIDIA/Cosmos-RL) の上に構築されています。

各ジョブは次で構成されます。

- モデルを学習する 1 個以上の **policy replica**
- 推論を実行して rollout sample を生成する 1 個以上の **rollout replica**

これらの component は中央の **cosmos-controller** によって協調されます。cosmos-controller は rollout の dispatch、reward の収集、training buffer の管理、最新 policy weight の rollout replica への定期同期を行います。この設計により、rollout generation と policy optimization を疎結合に保ちながら、大規模な非同期 RL 学習を可能にします。

RL アルゴリズムとして [GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) を使います。

<a id="key-files"></a>
### 主要ファイル

| パス                                                             | 目的                                                                                                                                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `toml/alpamayo_rvla_rl_local_test.toml`                          | Cosmos-RL TOML config (ローカルテスト)。replica 数、parallelism、optimizer、rollout、reward weight、logging、checkpointing を制御します。完全な schema は Cosmos-RL docs を参照してください。 |
| `models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py` | `cosmos-rl` に渡す entry script。model、rollout、trainer、reward を登録します                                                                            |
| `hydra_configs/alpamayo1_rvla_rl_pai.yaml`                       | PAI データセットと preprocessing 用の Hydra config (Alpamayo 1.5 では `alpamayo1_5_rvla_rl_pai.yaml` も参照)                                            |
| `launcher.py`                                                    | state を初期化し、Cosmos worker を呼び出す共通 launch logic                                                                                              |
| `rewards/aggregated_reward.py`                                   | デモ reward 実装 (下の [報酬](#reward) を参照)                                                                                                          |
| `../../scripts/convert_cosmos_rl_checkpoint.py`                  | Cosmos-RL policy checkpoint を HuggingFace checkpoint ディレクトリに変換します ([step 7](#7-export-the-rl-checkpoint-for-inference) を参照)              |

<a id="key-parameters"></a>
### 主要パラメータ

| パラメータ (TOML path)               | デフォルト (local test) | 意味                                                                                                                                   |
| ------------------------------------ | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `policy.parallelism.n_init_replicas`  | 1                       | policy replica 数。各 replica は独立した FSDP training worker です。replica が多いほど global batch が大きくなります。                 |
| `policy.parallelism.dp_shard_size`    | 4                       | policy replica あたりの GPU 数 (FSDP sharding degree)。`n_init_replicas × dp_shard_size` = policy GPU 総数。                           |
| `rollout.parallelism.n_init_replicas` | 1                       | rollout replica 数。各 replica は completion を生成する vLLM engine を実行します。policy consumption speed に合わせて scale します。    |
| `train.train_batch_per_replica`       | 48                      | policy replica ごと、step ごとに消費する training sample 数。**Global batch / step** = `policy.n_init_replicas × train_batch_per_replica`。 |
| `rollout.batch_size`                  | 2                       | 1 batch で rollout replica に送られる prompt 数。                                                                                       |
| `rollout.n_generation`                | 12                      | prompt ごとに生成する completion 数 (GRPO の "group")。各 prompt は reward で rank 付けされる `n_generation` 個の candidate rollout を生成します。 |
| `train.sync_weight_interval`          | 2                       | 最新 policy weight を rollout replica に同期する training step 間隔。小さいほど rollout は新鮮になりますが、通信 overhead は増えます。 |

**Dataloading acceleration.** Physical AI の training sample は大きいです。デフォルトの Cosmos-RL pipeline は、ノード上の GPU rank ごとに同じ sample を独立に読み込み preprocessing するため、I/O bandwidth と CPU time の両方を浪費します。node-level prefetch server (`prefetch/server.py`) は sample を事前に取得して preprocessing し、その結果を shared memory 経由で全 local rank と共有します。これにより、policy iteration の step time を大きく短縮できます (例: 44 秒 → 5 秒)。

| パラメータ (TOML path)                 | デフォルト (local test) | 意味                                                                                                                                                                  |
| -------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `custom.alpamayo.prefetch.capacity`    | 16                      | cache size (sample 数)。`train_batch_per_replica × replicas_per_node` に設定します。`<= 0` で prefetch を無効化します (rank ごとの同期的 dataset loading に fallback)。 |
| `custom.alpamayo.prefetch.num_workers` | 5                       | background で sample の取得と preprocessing を行う worker thread 数。                                                                                                  |

<a id="reward"></a>
### 報酬

同梱の reward function ([`rewards/aggregated_reward.py`](rewards/aggregated_reward.py)) は **デモ実装**です。各 rollout sample を 2 つの軸で評価し、1 つの scalar に結合します。

| コンポーネント | 計算方法                                                                        | TOML の weight key |
| -------------- | ------------------------------------------------------------------------------- | ------------------ |
| **ADE**        | 予測軌道と ground-truth 軌道 (XY) の Average Displacement Error (L2)            | `traj_l2_weight`   |
| **Comfort**    | comfort bounds (acceleration、jerk、yaw rate など) 内にある timestep の割合     | `comfort_weight`   |

この reward は gated structure を使います。ADE が threshold (default 3.0 m) を超える場合、reward は -1 に clamp されます。それ以外の場合は、正規化された ADE penalty と comfort score の重み付き結合になります。weight は TOML の `[custom.alpamayo.reward]` 配下で設定します。

> **Note:** この reward は出発点として提供されており、rollout の trajectory 部分のみを採点します。Alpamayo 1.5 は、generation の chain-of-causation 部分に reasoning reward を適用して RL post-training されました。同じ Cosmos-RL reward interface に従い、entry script に登録することで、trajectory と reasoning の両方を採点するものを含め、独自の reward を実装できます。rollout から reasoning text を抽出する方法の詳細は、下の FAQ *"reasoning (chain-of-thought) generation を RL で post-train できますか？"* を参照してください。

<a id="multi-node-large-scale-training"></a>
## マルチノード大規模学習

上記のローカルテストは単一ノードで実行します。マルチノードクラスタ学習へスケールするには、TOML の 2 つのパラメータを**必ず**変更してください。

1. **`policy.parallelism.dp_shard_size`** — 8 に設定します。これは FSDP sharding を制御します。`n_policy_replicas × dp_shard_size` = policy GPU 総数です。

2. **`train.train_policy.data_dispatch_as_rank_in_mesh`** — `true` に設定します。これにより rank-based data dispatch が有効になり、各 policy replica が dataset の安定した重複しない shard を消費します。これがないと、複数の replica が重複 sample で学習する可能性があります。この flag は data preloading の実行にも必要です。

また、特定の設定に合わせて次のパラメータを調整することをおすすめします。下の値は Alpamayo 1.5 の post-training で使用したものです。

| パラメータ                            | Local test | Cluster training |
| ------------------------------------- | ---------- | ---------------- |
| `policy.parallelism.n_init_replicas`  | 1          | 64               |
| `rollout.parallelism.n_init_replicas` | 1          | 128              |
| `train.train_batch_per_replica`       | 48         | 40               |
| `train.optm_lr`                       | 2e-6       | 2e-6             |
| `train.sync_weight_interval`          | 2          | 5                |
| `rollout.batch_size`                  | 2          | 6                |
| `rollout.n_generation`                | 12         | 12               |
| `custom.alpamayo.prefetch.capacity`   | 16         | 128              |

これにより、training step あたり 64 × 40 = **2560** sample の global batch になります。policy GPU は 512、rollout GPU は 128 です (合計 640 GPU、80 nodes)。SLURM の起動手順は [Cosmos-RL multi-node documentation](https://nvidia-cosmos.github.io/cosmos-rl/multinodes/overview.html) を参照してください。

> **Tip:** まず中規模 (例: policy replica 4、rollout replica 8) から始め、スケールアップ前に `controller.log` の `pending rollouts` を監視してください。tuning の目安は FAQ の *"policy replica と rollout replica の balance はどう取りますか？"* を参照してください。

<a id="faq"></a>
## FAQ

<details>
<summary><strong>具体的に何が RL post-training されますか？</strong></summary>

このコードは、公開済み Alpamayo モデル (ReasoningVLA) の **VLM backbone** を RL post-training します。この経路では、VLM が text と離散 trajectory token を autoregressive に生成します。action expert head (flow-matching ベースの continuous action) は、この RL pipeline では**学習されません**。**action expert pathway** の RL post-training は将来のリリースで提供予定です。

</details>

<details>
<summary><strong>Alpamayo 1.5 を post-train できますか？</strong></summary>

はい。[Alpamayo 1](https://huggingface.co/nvidia/Alpamayo-R1-10B) と [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) の両方をサポートしています。

変更が必要なのは 2 点です。

1. `ALPAMAYO_MODEL_DIR` を、使いたいモデルの変換済み checkpoint に向けます。
2. entry script で、モデルに合うように `hydra_config_name` を設定します。
   - Alpamayo 1.5: `"alpamayo1_5_rvla_rl_pai"`
   - Alpamayo 1: `"alpamayo1_rvla_rl_pai"`

</details>

<details>
<summary><strong>このコードで何ができますか？また RL checkpoint はどう使いますか？</strong></summary>

- **運転挙動の改善** — trajectory accuracy、comfort、safety、その他の driving metric を対象にした reward function を定義できます。
- **推論とシーン理解の改善** — モデル出力の text 部分を採点する reward を追加し、より良い状況認識と意思決定へモデルを誘導できます。
- **独自の driving data での学習** — PAI format の dataset を準備し、config をそこへ向け、自分の scenario で RL を実行できます。
- **RL checkpoint の使用** — export された checkpoint には **VLM backbone weights のみ**が含まれます (RL は VLM backbone のみを学習するため)。これは Alpamayo 1 と Alpamayo 1.5 の両方に読み込めます。Alpamayo 1.5 ディレクトリから使う場合は、checkpoint の model config で軽微な target renaming を行い、non-strict weight loading で SFT による action expert model の学習に使えます。action expert weights はランダム初期化される点に注意してください。

</details>

<details>
<summary><strong>reward function を置き換えるには？</strong></summary>

Cosmos-RL reward interface に従って独自 reward を実装し、entry script (`finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py`) に登録してください。期待される signature と return format は [`finetune/rl/rewards/aggregated_reward.py`](rewards/aggregated_reward.py) を参照してください。

</details>

<details>
<summary><strong>reasoning (chain-of-thought) generation を RL で post-train できますか？</strong></summary>

はい。モデルは `<|cot_end|>` の前に reasoning text を、`<|traj_future_start|>` の後に trajectory token を生成します。どちらも reward function に渡される単一の rollout completion string (`to_be_evaluated`) の一部です。現在の default reward (`aggregated_reward.py`) は trajectory 部分 (ADE + comfort) のみを採点しますが、reasoning trace も採点するように拡張できます。

rollout completion から reasoning text を抽出するには、次のようにします。

```python
reasoning_text = to_be_evaluated.split("<|cot_end|>")[0]
```

その後、custom reasoning reward (例: LLM-based grader、rule-based check、learned reward model) で採点できます。reasoning label と対応する reasoning reward function は近日公開予定です。

</details>

<details>
<summary><strong>推奨 GPU 数は？</strong></summary>

データセットサイズによって異なります。一般に、より大きな global batch size (`policy.parallelism.n_init_replicas` × `train.train_batch_per_replica`) は、より良い RL performance につながります。

おおまかな目安:

| Scale                  | Policy                                              | Rollout                           | `train_batch_per_replica` | `rollout.batch_size` × `n_generation` | Global batch / step |
| ---------------------- | --------------------------------------------------- | --------------------------------- | ------------------------- | ------------------------------------- | ------------------- |
| Local test (1 node)    | 4 GPUs, 1 replica, `dp_shard_size=4`                | 1 GPU, 1 replica                  | 48                        | 2 × 12 = 24                           | 48                  |
| Large scale (80 nodes) | 64 nodes (512 GPUs): 64 replicas, `dp_shard_size=8` | 16 nodes (128 GPUs): 128 replicas | 40                        | 6 × 12 = 72                           | 2560                |

- **Global batch / step** = `n_init_replicas` × `train_batch_per_replica`。
- **Policy GPUs** は学習速度と global batch size を決めます。モデルは FSDP (`dp_shard_size`) によって GPU 間で shard されます。
- **Rollout GPUs** は data generation throughput を決めます。policy consumption speed に合わせて rollout replica を scale してください。
- **rollout ≈ policy speed を保つ**: rollout が速すぎると data が stale になり、遅すぎると policy が idle になります。controller log の `pending rollouts` を監視してください。

</details>

<details>
<summary><strong>新しい reward、data、model に対する推奨 workflow は？</strong></summary>

1. **1 sample に overfit する。** 単一の training sample を作成し、ローカルで RL を実行します。reward が増加することを確認してください。これにより、reward function、data pipeline、model が正しく接続されていることを確認できます。

2. **1 ノードで小さな集合 (~16-32 samples) に overfit する。** 複数 epoch にわたって reward が改善することを確認します。この段階で reward weight、learning rate、`n_generation` を調整してください。rollout/policy speed imbalance に注意します (下の FAQ *"policy replica と rollout replica の balance はどう取りますか？"* を参照)。

3. **マルチノードへ scale する。** policy replica と rollout replica を増やします。`pending rollouts` と `weight_version` gap を監視し、system が balanced であることを確認します。中程度の global batch size (例: 320) から始め、reward variance が高すぎる場合は scale up します。

4. **reward function を反復改善する。** RL は reward が測定するものを最適化します。model behavior が期待どおりに改善しない場合は、さらに scale する前に reward design を見直してください。

</details>

<details>
<summary><strong>policy replica と rollout replica の balance はどう取りますか？</strong></summary>

rollout replica は非同期に data を生成し、policy replica はそれを学習に消費します。両者はおおむね同じ throughput で動く必要があります。

**Rollout が速すぎる (最も一般的):** 完了した rollout が controller buffer に積み上がります。policy がそれらで学習するころには、policy weight が rollout を生成した weight から大きく進んでいます (大きな `weight_version` gap → off-policy data → training quality の低下)。極端な場合、rollout worker はすべての epoch を終えているのに、training はまだ半分しか進んでいません。

**Rollout が遅すぎる:** policy が data 待ちで idle になり、GPU utilization が低下します。

**診断方法** — controller log で次の metric を確認します。

1. **`pending rollouts` が単調増加する** (rollout が速すぎる): buffer が drain されません。
2. **Rollout が早期終了する**: training が `total_steps` から遠い段階で `[Controller] All rollouts have ended` が表示されます。
3. **`pending rollouts` が頻繁に 0 まで落ちる** (rollout が遅すぎる): policy が次の rollout batch を待っています。

**例 — rollout が速すぎる場合:**

```text
# controller.log — buffer grows every step, never drains
Stat: samples=  24  pending=   24          ← start
Stat: samples= 600  pending=  264          ← growing
Stat: samples=1200  pending=  552          ← still growing
Stat: samples=2400  pending=  984          ← rollout outpacing policy
[Controller] All rollouts have ended … 1104 remaining rollouts
# training is at step 37/60 — 23 more steps will use stale rollouts
```

**例 — balance が取れている場合:**

```text
# controller.log — buffer stays small, oscillates
Stat: samples=  48  pending=   24
Stat: samples= 600  pending=   48          ← buffer stays low
Stat: samples=1200  pending=   72
Stat: samples=2400  pending=   48          ← not accumulating
```

**調整ノブ (試しやすい順):**

| ノブ                                          | 効果                                                  | 使う場面                                      |
| --------------------------------------------- | ----------------------------------------------------- | --------------------------------------------- |
| prefetch を有効化 (`prefetch.capacity > 0`)   | policy iteration time を短縮 (例: 44 秒 → 12 秒)      | 常に推奨。単体で最も効果が大きい              |
| `rollout.batch_size` または `n_generation` を減らす | rollout throughput を下げる                           | rollout が policy よりかなり速い場合          |
| rollout replica を追加                        | rollout throughput を上げる                           | policy が data 待ちで idle になる場合         |
| policy replica (`n_init_replicas`) を追加      | policy を高速化する (より並列な学習)                  | 大規模で rollout が policy を上回る場合       |
| `dp_shard_size` を増やす                      | より多い data parallelism で step ごとの学習を高速化  | 各 step が遅すぎる場合                        |
| `epoch` を減らす、または `max_num_steps` を設定する | 生成すべき rollout 総数を減らす                       | rollout が training よりかなり早く終わる場合  |

**目標状態:** `pending rollouts` がほぼ安定している状態です。健全な大規模ジョブ (policy replica 64、rollout replica 128、global batch = 2560) では、buffer は通常 global batch size の 4 倍程度を保持します。各 training batch 内の `weight_version` gap は、`sync_weight_interval` の数倍以内に収まるべきです。

</details>
