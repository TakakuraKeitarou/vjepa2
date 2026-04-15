# V-JEPA 2.1 評価ツール

学習済み V-JEPA 2.1 世界モデルの性能を検証するための評価・可視化ツール群です。

## ファイル構成

```
eval/
├── README.md                # このファイル
├── eval_world_model.py      # 未来予測Loss評価
├── vis_attention.py         # Attention Rollout 可視化
└── vis_tsne.py              # t-SNE 特徴量クラスタリング
```

## 前提条件

- conda 環境 `vjepa2-312` がアクティベートされていること
- 学習済みチェックポイント（`best.pth.tar` 等）が存在すること
- 評価用CSVファイル（`autodriving_val_paths.csv` 等）が存在すること
- 追加パッケージ: `matplotlib`, `scikit-learn`, `opencv-python`

```bash
conda activate vjepa2-312
pip install matplotlib scikit-learn
```

## 実行方法

すべてのスクリプトは **プロジェクトルート** (`vjepa2/`) から実行してください。

### 共通引数

| 引数 | 説明 | 例 |
|---|---|---|
| `--fname` | 学習時に使用したYAML設定ファイル | `configs/train_2_1/vitg16/autodriving-future-256px.yaml` |
| `--ckpt` | 学習済みチェックポイント | `runs/.../best.pth.tar` |
| `--val_csv` | 評価用動画リストCSV | `configs/.../autodriving_val_paths.csv` |
| `--device` | 使用GPU（デフォルト: `cuda:0`） | `cuda:1` |

---

### 1. 未来予測 Loss 評価 (`eval_world_model.py`)

未知の検証動画に対して、モデルの未来予測精度（MAE Loss）を定量的に測定します。

```bash
python eval/eval_world_model.py \
  --fname configs/train_2_1/vitg16/autodriving-future-256px.yaml \
  --ckpt runs/vjepa_2_1/autodriving_future/<日時フォルダ>/best.pth.tar \
  --val_csv configs/train_2_1/vitg16/autodriving_val_paths.csv
```

**出力**: 標準出力にバッチごとの Loss と最終的な Average Loss を表示

**何が分かるか**: Predictor（未来予測モジュール）の精度。値が低いほど未来予測が正確。

---

### 2. Attention Rollout 可視化 (`vis_attention.py`)

Target Encoder が映像のどの領域に注目しているかを、Attention Rollout 手法でヒートマップ化します。

```bash
python eval/vis_attention.py \
  --fname configs/train_2_1/vitg16/autodriving-future-256px.yaml \
  --ckpt runs/vjepa_2_1/autodriving_future/<日時フォルダ>/best.pth.tar \
  --val_csv configs/train_2_1/vitg16/autodriving_val_paths.csv \
  --num_samples 10 \
  --output_dir attention_results
```

| 固有引数 | 説明 | デフォルト |
|---|---|---|
| `--num_samples` | 可視化するサンプル数 | `10` |
| `--output_dir` | 出力先ディレクトリ | `attention_results` |

**出力**: `attention_results/attention_000.png` 〜 `attention_009.png`

各画像には4フレーム分の元映像（上段）とAttention Rollout ヒートマップ（下段）が並びます。

**何が分かるか**: エンコーダの特徴抽出品質。車両・信号・標識など自動運転に重要な物体に注目できているかを確認できます。

**手法の詳細**:
- 全40層の Transformer Block から QKV 出力を Hook で取得
- 各層で明示的に Attention 行列を計算（SDPAを無効化）
- 残差接続を考慮した Attention Rollout（`0.5*I + 0.5*A` を全層で累積）により、端部アーティファクトを軽減

---

### 3. t-SNE 特徴量クラスタリング (`vis_tsne.py`)

多数の検証動画から抽出した特徴量を2次元に圧縮し、モデルが異なるシーンをどのように分類しているかを散布図で可視化します。

```bash
python eval/vis_tsne.py \
  --fname configs/train_2_1/vitg16/autodriving-future-256px.yaml \
  --ckpt runs/vjepa_2_1/autodriving_future/<日時フォルダ>/best.pth.tar \
  --val_csv configs/train_2_1/vitg16/autodriving_val_paths.csv \
  --max_samples 300 \
  --output tsne_cluster.png
```

| 固有引数 | 説明 | デフォルト |
|---|---|---|
| `--max_samples` | 解析する最大動画数 | `300` |
| `--output` | 出力ファイルパス | `tsne_cluster.png` |

**出力**: `tsne_cluster.png`（散布図）

**何が分かるか**: エンコーダの表現学習品質。点の塊（クラスタ）が形成されていれば、モデルがシーンの種類を内部的に区別できている証拠です。

**手法の詳細**:
- Target Encoder で各動画の特徴量 (B, N, 1408) を取得
- Global Average Pooling で 1動画 → 1ベクトル (1408次元) に圧縮
- PCA (50次元) → t-SNE (2次元) の2段階次元削減

---

## 各ツールで分かることの対応表

| ツール | 将来予測の精度 | エンコーダの品質 |
|---|---|---|
| `eval_world_model.py` | **直接測定** | 間接的 |
| `vis_attention.py` | 間接的 | **直接確認** |
| `vis_tsne.py` | 間接的 | **直接確認** |

3つの結果を総合して「V-JEPA が自動運転映像から世界の物理法則を獲得できたか」を判断します。
