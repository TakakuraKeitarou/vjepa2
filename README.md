# V-JEPA 2: 動画モデルを用いた理解・予測・計画 (日本語版・自動運転拡張版)

🆕 **[2026-03-16]:** 🔥 V-JEPA 2.1 がリリースされました 🔥 より高品質で時間的整合性の取れた密な特徴量を学習できる新しいレシピを採用しています！

---

## 🚗 本プロジェクトでの独自の変更点（自動運転の世界モデル構築）

本リポジトリでは、オリジナルのV-JEPA 2.1を拡張し、**「自動運転データから未来の映像特徴量を予測する世界モデル（World Model）」** を構築するための以下の独自機能・設定が追加されています。

### 追加された主な機能とファイル
1. **自動データセットフォーマッター (`generate_dataset_csvs.py`)**
   - 大量の動画が入ったフォルダ（`train` / `val` / `eval`）を自動スキャンし、V-JEPAが読み込める形式のCSVファイル（例: `autodriving_train_paths.csv`）を一瞬で自動生成します。
   - 単一の動画フォルダを指定した場合でも、指定した比率（例: `--ratio 0.8 0.1 0.1`）に従って自動的にランダム分割・シャッフルを行う機能も搭載しています。
2. **自動運転・未来予測用YAML設定 (`configs/train_2_1/vitg16/autodriving-future-256px.yaml`)**
   - V-JEPA 2.1の純粋な機能のみを用いて、「映像の前半75%」だけを文脈としてエンコーダーに渡し、「残りの25%（未来）」のみを予測ターゲットとして学習する構成になっています（アクションデータは不要）。
   - エポック数、バッチサイズ（デフォルト: 4）、学習率のスケジュールなどのハイパーパラメータを調整済みで、すぐに学習にかけることができます。
3. **TensorBoardへのリアルタイムログ出力**
   - `app/vjepa_2_1/train.py` にパッチを当て、標準のCSV出力に加えて、自動的にTensorBoardへLossと学習率（LR）のグラフを描画する機能を追加実装しました。
4. **学習の安全性向上・拡張機能**
   - 学習を実行するたびに自動的に「実行日時のフォルダ（例: `20260414_1645`）」が作成されるようになり、過去の学習ファイルが意図せず上書きされるのを防ぎます。
   - すべてのエポックの中で「最もLossが低かった最高の成績の重み」を自動的に `best.pth.tar` として特別保存する機能を追加しました。

---

## 🔧 環境構築

### 必要なライブラリ
事前に以下をインストールしてください。CUDAサポート付きのPyTorchを強く推奨します。
- [PyTorch](https://pytorch.org/get-started/locally/)
- [timm](https://pypi.org/project/timm/)
- [einops](https://pypi.org/project/einops/)

### セットアップ

```bash
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # 開発モードの場合は pip install -e .
```

### ⚠️ macOSユーザーへの注意

V-JEPA 2 は動画読み込みに [`decord`](https://github.com/dmlc/decord) を使用していますが、現在 macOS には対応していません。macOS で動かす場合は、代替実装として [`eva-decord`](https://github.com/georgia-tech-db/eva-decord) や [`decord2`](https://github.com/johnnynunez/decord2) の使用が報告されています。いずれを選択するかはユーザーの判断に委ねられています。

---

## 🛠 クイックスタート（世界モデルの学習方法）

### 1. データの準備
お手元にある `.mp4` などの自動運転動画データセットから、学習用CSVリストを生成します。

```bash
# 仮想環境を有効化
conda activate vjepa2-312

# フォルダ内にある動画をスキャンし、80:10:10の割合でTrain/Val/Eval用のCSVを作成
python generate_dataset_csvs.py --data_dir /path/to/your/videos --ratio 0.8 0.1 0.1
```
※実行すると `configs/train_2_1/vitg16/` 以下にCSVファイルが生成されます。

### 2. 学習の実行（マルチGPU対応）
生成したCSVを用いたYAML設定ファイルを指定し、学習プロセスを開始します。
（内部で `Distributed Data Parallel` による通信設定が組まれているため、GPUを複数枚持っている場合は `--devices` に続けて追加するだけで並列学習されます）

```bash
# 例: 3枚のGPU (RTX PRO 6000 x3など) を利用して並列学習を進める場合
NCCL_SOCKET_IFNAME=lo python -m app.main --fname configs/train_2_1/vitg16/autodriving-future-256px.yaml --devices cuda:0 cuda:1 cuda:2
```
*※毎回、自動的に `./runs/vjepa_2_1/autodriving_future/YYYYMMDD_HHMMSS/` という日時フォルダが切られて学習が始まります。*
*※ `NCCL_SOCKET_IFNAME=lo` は、ローカル通信のバグを回避して安定的にNCCLを動作させるためのおまじないです。*

### 3. TensorBoardでの監視
学習中（Loss）などの学習状況はリアルタイムに監視できます。
別のターミナルを開き、上で作成された新しい日時フォルダを指定して、以下のコマンドを打ってブラウザで `http://localhost:6006` を開いてください。

```bash
conda activate vjepa2-312
tensorboard --logdir ./runs/vjepa_2_1/autodriving_future/【ここを実行日時のフォルダ名にする】/tensorboard
```

### 4. 学習の再開（特定のフォルダからのレジューム機能）
一度 `Ctrl + C` 等で学習を完全に停止した後、途中から再開したい場合は `--resume_dir` という専用オプションで**続きから始めたい過去の日時フォルダを直接指定**して実行してください。

```bash
NCCL_SOCKET_IFNAME=lo python -m app.main --fname configs/train_2_1/vitg16/autodriving-future-256px.yaml --devices cuda:0 cuda:1 cuda:2 --resume_dir ./runs/vjepa_2_1/autodriving_future/20260414_164530
```

これだけで指定したフォルダ内の `latest.pth.tar` を自動検出し、エポック数や学習スケジュールなども完璧に復元された状態で同じフォルダ内に続きから上書き再開されます。

---

## オリジナル版 V-JEPA 2 概要

V-JEPA 2 は、インターネット上の大規模な動画データを用いて、物理世界の理解や予測を自己教師あり学習（ラベル無し）によって習得する最新モデルです。
* **V-JEPA 2:** 行動予測やモーション理解においてState-of-the-Artを達成。
* **V-JEPA 2-AC:** V-JEPA 2をベースに、少量のロボットデータを追加学習させることで、環境に依存せず多様なマニピュレーションタスク（リーチ・把持・移動など）をゼロからこなせるようにしたモデル。

### モデルの構造
* **Dense Predictive Loss (密な予測損失):** 全てのトークン（情報を与える文脈側と、予測すべきマスク側の両方）が自己教師ありの誤差逆伝播に貢献する新しい方式を採用。
* **Deep Self-Supervision:** エンコーダの途中段階の表現レイヤーに対しても損失計算を適用することで、抽象度の異なる特徴をうまく学習できます。

### 公式の事前学習済みモデル（V-JEPA 2.1）
解像度384px に対応した各サイズの事前学習済みチェックポイントも公開されています。
- `ViT-B/16 (80M)`
- `ViT-L/16 (300M)`
- `ViT-g/16 (1B)`
- `ViT-G/16 (2B)`

詳細なモデルのダウンロード用リンクや論文については、オリジナルのリポジトリ（メタ社の公式GitHub）をご参照ください。

---
