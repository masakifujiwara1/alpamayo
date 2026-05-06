<div align="center">

# 🏔️ Alpamayo 1

### 汎化可能な自動運転に向けて、推論と行動予測をつなぐ

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

## 更新情報

- [2026 年 4 月] ⚙️ [ファインチューニングスクリプト](#ファインチューニングスクリプト)を公開しました。教師ありファインチューニング向けの [SFT](docs/FINETUNE_SFT.md) と、強化学習ベースの post-training 向けの [RL](finetune/rl/README.md) が含まれます。
- [2026 年 3 月] [🏔️ Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5) が公開されました。性能改善、新機能、継続サポートのため、すべてのユーザーに新バージョンの確認をおすすめします。🚀
- [2026 年 1 月] CES 2026 での [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) 公開に伴い、Alpamayo-R1 は Alpamayo 1 に改名されました。

______________________________________________________________________

**📖 まず [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) をお読みください。**
モデルカードには、モデルアーキテクチャ、入力/出力、ライセンス、検証済みハードウェア構成に関する包括的な詳細が記載されています。この GitHub README は、セットアップ、使用方法、よくある質問に焦点を当てています。

## 要件

| 要件       | 仕様                                                                 |
| ---------- | -------------------------------------------------------------------- |
| **Python** | 3.12.x (`pyproject.toml` を参照)                                     |
| **GPU**    | 24 GB 以上の VRAM を持つ NVIDIA GPU (例: RTX 3090, RTX 4090, A5000, H100) |
| **OS**     | Linux (検証済み)。その他のプラットフォームは未検証                  |

> ⚠️ **注意**: VRAM が 24 GB 未満の GPU では、CUDA out-of-memory エラーが発生する可能性が高いです。

## インストール

### 1. uv のインストール (未インストールの場合)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. 環境のセットアップ

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 3. HuggingFace 認証

モデルは gated resource へのアクセスを必要とします。以下からアクセスを申請してください。

- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

その後、HuggingFace CLI で認証します。

```bash
pip install -U huggingface_hub
hf auth login
```

アクセストークンはこちらで取得できます: https://huggingface.co/settings/tokens

> 💡 **Tip**: HuggingFace 認証の詳細は[公式ドキュメント](https://huggingface.co/docs/huggingface_hub/guides/cli)を参照してください。

## 推論の実行

### テストスクリプト

注意: このスクリプトは、いくつかのサンプルデータ (比較的小さい) とモデル重み (22 GB) の両方をダウンロードします。
後者はネットワーク帯域によって特に時間がかかる場合があります。
参考として、100 MB/s の有線接続では約 2.5 分かかります。

```bash
python src/alpamayo_r1/test_inference.py
```

より多くの軌道や推論トレースを取得したい場合は、`num_traj_samples=1` 引数をより大きい値に変更してください (60 行目)。

### インタラクティブノートブック

同様の推論コードを含むノートブックを `notebook/inference.ipynb` に用意しています。

## 論文との関係

Alpamayo 1 は、論文 [*"Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail"*](https://arxiv.org/abs/2511.00088) で説明されているアーキテクチャを実装しています。内容は次のとおりです。

| 機能                                    | 論文での説明                                                     | このリリース (v1.0) |
| --------------------------------------- | ---------------------------------------------------------------- | ------------------- |
| **Chain-of-Causation (CoC) reasoning**  | 推論トレースのための human-in-the-loop なハイブリッド自動ラベリング | ✅ 含まれる          |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ 含まれる          |
| **Trajectory prediction**               | 6.4 秒 horizon、10 Hz で 64 waypoints                             | ✅ 含まれる          |
| **SFT fine-tuning (weights)**           | SFT 済みモデル重み                                               | ✅ 含まれる          |
| **SFT fine-tuning (code)**              | 教師ありファインチューニングパイプライン                         | ✅ 含まれる          |
| **RL post-training (weights)**          | RL post-training 済みモデル重み                                   | ❌ このリリースには含まれない |
| **RL post-training (code)**             | Cosmos-RL による RL post-training パイプライン                    | ✅ 含まれる          |
| **Route/navigation conditioning**       | 明示的な navigation または route 入力                             | ❌ このリリースには含まれない |
| **Meta-actions/General VQA**            | 高レベル行動と視覚質問応答                                       | ❌ このリリースには含まれない |

このリリースには、コアモデル、SFT スクリプト、RL post-training パイプラインが含まれます。RL 学習済み重み、route conditioning、meta-actions は将来リリースの候補です。

## ファインチューニングスクリプト

| 手法    | 説明                                                   | ドキュメント                      |
| ------- | ------------------------------------------------------ | --------------------------------- |
| **SFT** | 教師ありファインチューニング                           | [SFT guide](docs/FINETUNE_SFT.md) |
| **RL**  | Cosmos-RL による強化学習ベースの post-training         | [RL guide](finetune/rl/README.md) |

計算要件、手順、ファインチューニング FAQ については、リンク先のガイドを参照してください。

## よくある質問 (FAQ)

<details>
<summary><strong>10B モデルは navigation/route 入力を受け付けますか？</strong></summary>

route conditioning 機能の実験は行っていますが、公開モデルにはこの機能は含まれていません。現在のリリースは、waypoint や turn-by-turn navigation instruction のような明示的な navigation/route 入力なしで、マルチカメラ動画と egomotion 履歴を入力として受け取ります。

</details>

<details>
<summary><strong>モデルは meta-actions を生成しますか？または general VQA をサポートしますか？</strong></summary>

meta-action や general VQA 機能の実験は行っていますが、公開モデルにはこれらの機能は含まれていません。Alpamayo 1 は Chain-of-Causation reasoning を伴う軌道予測に特化して設計されており、軌道 + 推論トレースの出力を生成します。

</details>

<details>
<summary><strong>10B モデルは Reinforcement Learning (RL) で post-training されていますか？</strong></summary>

いいえ。現在公開されている 10B モデルは **RL post-training を受けていません**。論文では推論品質と行動一貫性を改善するための RL 段階を説明していますが、このリリースは教師あり学習コンポーネントに焦点を当てています。前述のとおり、将来のリリースで RL post-trained model を公開する可能性があります。

</details>

<details>
<summary><strong>最小 GPU 要件は何ですか？</strong></summary>

推論には少なくとも **24 GB VRAM** を持つ NVIDIA GPU が必要です。検証済み構成には RTX 3090、A100、H100 が含まれます。より少ないメモリの GPU (例: 16 GB) で実行すると、CUDA out-of-memory エラーになる可能性が高いです。

</details>

<details>
<summary><strong>このモデルを production / commercial application で使えますか？</strong></summary>

いいえ。モデル重みは **non-commercial license** で公開されています。このリリースは、研究、実験、評価目的のみを想定しています。詳細は [License](#ライセンス) セクションと [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) を参照してください。

</details>

## プロジェクト構成

```
alpamayo/
├── finetune/
│   ├── rl/                              # RL post-training
│   │   ├── models/                      # モデル wrapper と Cosmos-RL entry script
│   │   ├── rewards/                     # 報酬関数
│   │   ├── prefetch/                    # shared-memory data prefetch server
│   │   ├── toml/                        # Cosmos-RL 学習 config
│   │   ├── hydra_configs/               # dataset と preprocessing config
│   │   └── README.md                    # RL post-training ガイド
│   └── sft/                             # 教師ありファインチューニング
│       ├── configs/                     # モデル config
│       ├── models/                      # 学習可能 wrapper
│       ├── train_hf.py                  # 学習スクリプト
│       └── evaluate_hf.py               # 評価スクリプト
├── notebook/
│   └── inference.ipynb                  # サンプルノートブック
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # action space 定義
│       ├── diffusion/
│       │   └── ...                      # diffusion model component
│       ├── geometry/
│       │   └── ...                      # geometry utility と module
│       ├── models/
│       │   ├── ...                      # モデル component と utility function
│       ├── __init__.py                  # package marker
│       ├── config.py                    # モデルと実験の configuration
│       ├── helper.py                    # utility function
│       ├── load_physical_aiavdataset.py # dataset loader
│       ├── test_inference.py            # 推論テストスクリプト
├── pyproject.toml                       # project dependency
└── uv.lock                              # 固定された dependency version
```

## トラブルシューティング

### Flash Attention の問題

モデルは default で Flash Attention 2 を使います。互換性の問題が発生した場合は、次のように設定してください。

```python
# 代わりに PyTorch の scaled dot-product attention を使う
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory エラー

OOM エラーが発生した場合:

1. 少なくとも 24 GB VRAM を持つ GPU を使っていることを確認してください。
2. 複数の軌道を生成している場合は `num_traj_samples` を減らしてください。
3. GPU を多く使う他のアプリケーションを終了してください。

## ライセンス

- **推論コード**: Apache License 2.0。詳細は [LICENSE](./LICENSE) を参照してください。
- **モデル重み**: Non-commercial license。詳細は [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) を参照してください。

## 免責事項

Alpamayo 1 は、自動運転車 (AV) 領域の研究開発を加速するために設計された事前学習済み推論モデルです。自動運転の end-to-end backbone の構築から、推論ベースの自動ラベリングツールの実現まで、さまざまな AV 関連ユースケースの基盤として機能することを意図しています。つまり、カスタム AV アプリケーションを開発するための構成要素として捉えるべきです。

重要な注意:

- Alpamayo 1 は、研究、実験、評価目的のためにのみ提供されます。
- Alpamayo 1 は完全な driving stack ではありません。特に、実世界で重要なセンサー入力へのアクセスがなく、必要な多様かつ冗長な安全機構を組み込んでおらず、配備に向けた automotive-grade validation も受けていません。

このモデルを使用することにより、あなたはこれが科学的調査、ベンチマーク、探索を支援するための研究ツールであり、認証済み AV stack の代替ではないことを承認したものとみなされます。開発者およびコントリビューターは、モデルまたはその出力の使用について、いかなる責任も負いません。

## 引用

研究で Alpamayo 1 を使用する場合は、次を引用してください。

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
