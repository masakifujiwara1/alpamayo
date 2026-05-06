# Alpamayo R1 SFT

**以下の例は、各 80 GB の H100 GPU 8 基で検証されています。**

このガイドでは、Physical AI AV データセットを使って Alpamayo 1 モデルの教師ありファインチューニング (SFT) を実行する方法を説明します。SFT スクリプトは [finetune/sft](../finetune/sft/) にあります。

## 機能

- [x] Stage 1: ベース VLM のファインチューニング
- [x] Stage 2: エキスパート軌道拡散モデルのファインチューニング
- [x] データローダー対応
  - PAI: [physical_ai_av](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

## データセットとチェックポイントの準備

### PAI データセットのダウンロード (スクリプト)

完全なデータセット、チャンク単位のサブセット、または個別コンポーネントを必要に応じてダウンロードできます。

Alpamayo 1 用の代表的なスライス (ここではチャンク 0-10、4 台のカメラ、egomotion) をダウンロードするには、次のコマンドを使います。

先に Hugging Face トークンを設定してください。

`export HF_TOKEN=<your Hugging Face token>`

```
python scripts/download_pai.py --chunk-ids 0-10 --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov --calibration camera_intrinsics sensor_extrinsics --labels egomotion --output-dir /path/to/pai_dataset
```

> `--chunk-ids` は単一 ID (`0`)、複数 ID (`0 1`)、範囲 (`0-3`: チャンク 0、1、2 を取得) を受け付けます。省略するか `None` を渡すと、完全なデータセット (約 97 TB 規模) をダウンロードします。

### チェックポイントのダウンロード

事前学習済み Alpamayo 1 チェックポイントを [Hugging Face](https://huggingface.co/nvidia/Alpamayo-R1-10B) から **ローカルディレクトリ** にダウンロードします。Stage 1 は Hub のモデル ID だけでなく、ディスク上の重みを読み込みます。例:

```
huggingface-cli download nvidia/Alpamayo-R1-10B --local-dir <path/to/model>
```

学習時にそのディレクトリを参照させます。[ar1_base.yaml](../finetune/sft/configs/models/ar1_base.yaml) の `checkpoint_path` を設定するか、Stage 1 起動時にコマンドラインで `model.checkpoint_path=<path>` を渡してください (下記参照)。

## Stage 1 ファインチューニングの実行

> Alpamayo 1 は Hydra を使うため、構造化された形で設定を拡張または上書きできます。

> **Weights & Biases:** 実行ログを W&B に送るには、[sft_base.yaml](../finetune/sft/configs/sft_base.yaml) で `wandb` の default が有効になっていることを確認し (基本的には先頭付近の `wandb` 行のコメントを外します)、[wandb/default.yaml](../finetune/sft/configs/wandb/default.yaml) に [wandb.ai](https://wandb.ai) の `team` と `project` を入力します。必要なら `trainer` 配下の `report_to: wandb` を設定し、学習開始時に W&B API キーが使える状態にしてください。

### データローダー

PAI をダウンロードしたら、Hydra 設定または override で `local_dir` をデータセットのルート (例: `/path/to/pai_dataset`) に、`chunk_ids` を `"0-10"` のような範囲文字列に設定します。

### 学習の開始

学習は、収束性と安定性のために 2 段階のパイプラインを使います。

1. **Stage 1:** VLM (`base_model`) をファインチューニングし、離散軌道トークンを出力させます。
2. **Stage 2:** Stage 1 の VLM を凍結し、連続軌道用の action expert (trajectory diffusion) を学習します。

### ハイパーパラメータ

必要に応じて、`dataloader_num_workers` や学習率などの設定を config で調整できます。

#### Stage 1

> Stage 1 は VLM 全体をファインチューニングします。メモリ効率のよいマルチ GPU 学習のため、同梱 config と同様に DeepSpeed を有効にしてください。

Stage 1 では離散トークン用に [base_model.py](../src/alpamayo_r1/models/base_model.py) を学習します。

```
torchrun --nproc_per_node 8 -m finetune.sft.train_hf --config-path pkg://finetune/sft/configs --config-name sft_stage1
```

たとえば、チェックポイントの場所は次のように上書きできます。

`model.checkpoint_path=<path/to/Alpamayo-R1-10B>`

これは `huggingface-cli download nvidia/Alpamayo-R1-10B ...` で作成したディレクトリと同じである必要があります。

ログ出力例:

```
{'loss': 1.668, 'grad_norm': 0.9367678165435791, 'learning_rate': 1.2500000000000003e-08, 'epoch': 0.02}
{'loss': 1.7079, 'grad_norm': 1.1734423637390137, 'learning_rate': 2.5000000000000005e-08, 'epoch': 0.03}
{'loss': 1.641, 'grad_norm': 0.8667469620704651, 'learning_rate': 3.7500000000000005e-08, 'epoch': 0.05}
{'loss': 1.6859, 'grad_norm': 0.8352743983268738, 'learning_rate': 5.000000000000001e-08, 'epoch': 0.06}
{'loss': 1.6968, 'grad_norm': 1.1325007677078247, 'learning_rate': 6.250000000000001e-08, 'epoch': 0.08}
```

#### Stage 2

Stage 2 では trajectory diffusion expert を追加し、Stage 1 の VLM は凍結したままにします。

```
torchrun --nproc_per_node 8 -m finetune.sft.train_hf --config-path pkg://finetune/sft/configs --config-name sft_stage2 model.pretrained_model_name_or_path=/path/to/Alpamayo-R1-10B model.stage1_vlm_checkpoint_path=/path/to/stage1/output/checkpoint-xxxx
```

> `model.pretrained_model_name_or_path` は Stage 1 で使用したローカルフォルダ (ディスク上の完全なベースチェックポイント) と同じである必要があります。`model.stage1_vlm_checkpoint_path` は Stage 1 Trainer の出力です。例: `output_stage1/checkpoint-3500` (`model.safetensors.index.json` と shard を含むディレクトリ)。

次のような loss curve が表示されます。

![loss.png](loss.png)

### 評価

このコマンドは、config の `val_dataset` に対して Stage 2 チェックポイントを評価します。

```
torchrun --nproc_per_node 8 -m finetune.sft.evaluate_hf --config-path pkg://finetune/sft/configs --config-name sft_stage2 evaluate.eval_ckpt=/path/to/stage2/output/ckpt-xxx
```

上記の default 設定では、`val/metric/min_ade` は 1 未満になるはずです。メトリクス例:

```
val/metric/ade              2.0072
val/metric/ade/by_t=3.0     0.3970
val/metric/corner_distance  0.6632
val/metric/min_ade          0.6270
val/metric/min_ade/by_t=0.5 0.0079
val/metric/min_ade/by_t=1.0 0.0261
val/metric/min_ade/by_t=3.0 0.2008
val/metric/min_ade/by_t=5.0 0.4351
```

## サンプルメトリクスと loss curve に関する注意

このガイドの数値とプロットは、**検証と比較のためだけ**に示しています。リリースモデルはすでにこのデータで学習済みであるため、同じファインチューニング手順を実行しても、スクラッチ学習のような大きな loss 低下は見られません。これらの参照値は、大きな事前学習風の loss 低下を再現するためではなく、セットアップが**典型的な**ファインチューニング実行 (ログの形、メトリクスの大きさ、全体的な挙動) と合っていることを確認するために含めています。
