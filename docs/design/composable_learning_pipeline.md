# Composable 学習パイプライン設計（拡張方針）

## 目的

特徴量作成・スペクトル前処理・目的変数変換・データ拡張など、**適用タイミングが異なる処理**を、単一のパイプラインとして組み立てつつ、**拡張性と正しさ（CV / 推論の整合）**を両立できるようにするための設計指針をまとめる。

本稿の「Composable」は、`src/preprocess/core/base.py` の `ComposablePreprocessor` および `PreprocessingPipeline` が提供する **`>>` 連結可能な前処理チェーン**を指す。  
（別物として、`src/pipeline/` の MLflow 用 `Step` 連結も「パイプライン」と呼ばれるが、責務が異なるので混同しない。）

---

## 現状の分離（ベースライン）

本プロジェクトではすでに次のように責務が分かれている。

| 領域 | 主な API | 学習時 | 推論時 |
|------|-----------|--------|--------|
| **X（スペクトル／特徴量）** | `BasePreprocessor` の `fit` / `transform` / `fit_transform` | `fit_transform(train)` で学習＋適用 | `transform(test)` で同じ変換を適用 |
| **y（目的変数）** | `BaseTargetTransformer`（`src/model/target_transform.py`） | `fit` + `transform(y_train)` | モデル出力に `inverse_transform` |
| **実行オーケストレーション** | `CVTrainer` / `FullTrainer`、`PredictTest` | CV fold ごとに前処理を **train に fit**、valid は **transform のみ** | 学習済み `preprocessor.transform` → `model.predict` → `inverse_transform` |

参照: `PredictTest` は `preprocessor.transform` の後に予測し、必要なら `target_transformer.inverse_transform` をかける（`src/pipeline/steps/model_steps.py`）。

この分離は維持したまま、「データ拡張」のような **train 専用**の振る舞いをどう組み込むかが設計の焦点になる。

---

## 処理の 3 カテゴリ（契約）

将来の「単一インターフェイス」を語るとき、実際には **契約（contract）が異なる 3 種類のステップ**を区別する。

### 1. 対称ステップ（Symmetric / X-only）

**意味**: train・valid・test のいずれにも **同じ `transform` 意味**で適用する。

- 例: SNV、Savitzky–Golay、区間統計特徴、`FeatureUnion` の各 branch など。
- **インターフェイス**: 既存の `BasePreprocessor` で十分。
- **fit の対象**: 通常は **各 CV fold の train のみ**（リーク回避）。実装は `CVTrainer` が `fit_transform(train_ds)` / `transform(valid_ds)` で保証している。

### 2. 目的変数ステップ（Target / y）

**意味**: X ではなく **スカラー y** のスケール変換。推論では **必ず逆変換**が必要。

- **インターフェイス**: `fit(y)`、`transform(y)`、`inverse_transform(y_pred)`（既存の `BaseTargetTransformer`）。
- **パイプライン内の位置**: X の `PreprocessingPipeline` とは **別オブジェクト**のままが明確で、トレーナが「X パイプライン → モデル → y 逆変換」と順序保証するのが読みやすい。

### 3. 学習専用拡張ステップ（Train-only / Augmentation）

**意味**: **行を増やす・乱数で摂動する**など、学習データの分布を意図的に変える。valid / test では **適用しない**（または恒等）。

- **要求される振る舞い**:
  - **CV の train fold**: 拡張を適用したい（`fit_transform` 側で行数が増えてもよい）。
  - **CV の valid fold**: 拡張しない（元のスペクトルと y の対応を保つ）。
  - **test / submission**: 拡張しない。
- **既存 API との整合**: `CVTrainer` は train に `preprocessor.fit_transform(train_ds)`、valid に `preprocessor.transform(valid_ds)` を呼ぶ。したがって、拡張ステップは **`fit_transform` では拡張、`transform` では恒等（行数・内容ともに valid/test と整合）** と実装すれば、**呼び出し側を変えずに**挿入できる。

---

## 推奨アーキテクチャ: レイヤー + ステップ種別

「すべてを 1 つの巨大 Protocol に押し込む」のではなく、次の **2 層**で考えると拡張しやすい。

### レイヤー A: 学習パイプライン（`LearningPipeline` という概念）

1 回の学習・推論で確定すべき **契約の束**を表す。

| 構成要素 | 役割 |
|----------|------|
| `x_pipeline` | `BasePreprocessor`（単体または `PreprocessingPipeline`）。対称ステップと **train-only ステップを順序どおり**に含められる。 |
| `target_transformer` | `BaseTargetTransformer`。 |
| （任意）`metadata` | 乱数シード、拡張倍率など、再現性用。 |

**推論時の実行順**（現在の `PredictTest` と整合）:

```text
test Dataset
  -> x_pipeline.transform
  -> model.predict
  -> target_transformer.inverse_transform（必要時）
```

**学習時の実行順**（fold 内）:

```text
train Dataset
  -> x_pipeline.fit_transform   # ここで train-only ステップが拡張してよい
  -> target_transformer.fit(train の生 y); transform(y) で学習用 y
valid Dataset
  -> x_pipeline.transform       # train-only は恒等
  -> model.predict / OOF 収集
```

この「束」を `Recipe` が名前で指すか、専用の `LearningRecipe` を増やすかは実装時の選択（後述）。

### レイヤー B: ステップの「種別」タグ（概念的）

コード上は **クラス名・継承・または `StepKind` 列挙**で表現する。単一の巨大 `transform(ds, mode=...)` より、**振る舞いの違いを型で区別**した方がテストしやすい。

| 種別 | 名前例 | `fit_transform` | `transform` |
|------|--------|-------------------|-------------|
| 対称 | `SNVPreprocessor` | 学習して適用 | 同じ変換を適用 |
| train-only | `SpectralAugmentationTrainOnly` | 拡張後の `Dataset` を返す | 入力をそのまま返す（行増やさない） |

**重要**: `PreprocessingPipeline` はステップを順に適用する。train-only を **先頭付近**（生吸光度付近）に置くか、SNV 後に置くかはドメイン依存だが、**後続ステップは拡張後の行数に対して fit される**点に注意する。

---

## 単一インターフェイスを求める場合の選択肢

要件「すべてを 1 つのパイプライン形状で扱いたい」に対して、実務的には次の 3 パターンがある。

### 案 A: 現状維持 + train-only ステップ（推奨の最小拡張）

- X はこれまで通り `BasePreprocessor` の連鎖。
- 拡張だけ「`fit_transform` ≠ `transform` の意味」という **契約違い**をもつ 1 クラスとして追加する。
- y は `BaseTargetTransformer` を別インスタンスで保持。
- **長所**: `CVTrainer` / `PredictTest` の変更が最小。  
- **短所**: 「y は X パイプラインに入っていない」単一オブジェクトにはならない（ただし責務は明確）。

### 案 B: 実行コンテキスト付き `transform(ds, ctx)`

- すべてのステップに `FitContext`（例: `TRAIN_FIT` / `EVAL` / `INFERENCE`）を渡す。
- train-only は `ctx` に応じて拡張のオンオフ。
- **長所**: 将来、「calibration 用の別パス」などへ一般化しやすい。  
- **短所**: 既存の全ステップのシグネチャ変更が大きい。段階的導入にはアダプタが必要。

### 案 C: 推論用パイプラインの二重定義（非推奨になりがち）

- `train_pipeline` と `inference_pipeline` を別々にビルド。
- **短所**: レシピの二重管理・ドリフトのリスクが高い。

**推奨**: まず **案 A** で train-only ステップを追加し、必要になったら **案 B** を「新 Protocol + 旧クラス用アダプタ」で段階導入する。

---

## データ拡張を入れるときの必須ルール（本リポジトリ固有）

1. **`sample_id` の一意性**  
   `CVTrainer` は `sample_id` で OOF 集計用のインデックスを引く。拡張で行を増やす場合、**元と衝突しない `sample_id`** を付与する（例: 元 ID に派生サフィックスを付ける規約）。

2. **`groups` の一貫性**  
   Group CV では、拡張行は **元サンプルと同じ `groups`** を継承する。

3. **valid / test は恒等**  
   上記のとおり、拡張は `fit_transform` のみ、`transform` では行数を変えない。

---

## `Recipe`・JSON 設定との関係

- 現状: `Recipe` は `preprocessor_name` + `model_name` + `target_transform_name` を持つ（`src/recipes/base.py`）。
- 拡張後も、**「X チェーン」「y 変換」「モデル」**の 3 参照は分離のままが安全。  
- `preprocess_params.json` の `pipeline` 型で、train-only ステップを **先頭または明示された位置**に差し込む形が自然。

将来的に「学習パイプライン全体」を 1 名前で指したい場合は、`LearningPipelineSpec` のような設定ブロックを新設し、その中で `x_steps` / `target` / `seed` をまとめてもよい（実装は別タスク）。

---

## 既存コードとの対応表

| 概念 | 主な実装ファイル |
|------|------------------|
| X 前処理の連結 | `src/preprocess/core/pipeline.py`（`PreprocessingPipeline`） |
| 前処理 Protocol | `src/preprocess/core/base.py`（`BasePreprocessor`） |
| y 変換 | `src/model/target_transform.py` |
| fold 内の fit/transform 使い分け | `src/model/trainers/cv_trainer.py` |
| 推論時の逆変換 | `src/pipeline/steps/model_steps.py`（`PredictTest`） |

---

## まとめ

- **単一インターフェイス**とは、1 クラスにすべてを詰めることではなく、**「X パイプライン」「y 変換」「モデル」という契約の束**と、各ステップの **種別（対称 / train-only）** を明示する設計が拡張しやすい。
- **データ拡張**は `fit_transform` と `transform` の意味差で表現すると、既存の `CVTrainer`・`PredictTest` と整合しやすい。
- **y の逆変換**は引き続き `BaseTargetTransformer` に寄せ、推論の最後で適用するのが最も追いやすい。

この方針を前提に、具体的な `SpectralAugmentation` クラスや `PreprocessorBuilder` への登録は別途実装する。
