# Architecture Overview

## 概要

このプロジェクトは、木材の近赤外スペクトルから含水率を予測するための機械学習基盤です。  
設計上は、次の 4 つの責務を分離することを重視しています。

- データ表現
- 前処理とモデル条件の定義
- 学習 / 評価 / 予測の実行
- 実行インターフェース（CLI, MLflow）

これにより、前処理やモデルを差し替えても、CV 学習と提出用予測を同じ枠組みで再利用できます。

## 設計方針

このプロジェクトの全体設計は、次の考え方に基づいています。

- `Dataset` を中心にデータの入出力を統一する
- 前処理とモデル条件は `Recipe` と JSON 設定に分離する
- 実行処理は小さな `Step` に分解し、`>>` で `Pipeline` を構築する
- CV 学習と全件学習で同じ recipe を使い、条件のずれを防ぐ
- 学習結果や評価値は `State` に集約し、Step 間で受け渡す

## フォルダ構成

主要なディレクトリは次の通りです。

```text
docs/
  architecture.md
  data_info.md
  knowledge.md
  design/
    pipeline.md
    preprocess.md
    recipe.md

src/
  cli.py
  data/
  model/
  pipeline/
  preprocess/
    core/
    transforms/
    features/
  recipes/

tests/
  data/
  model/
  pipeline/
  preprocess/
  recipes/
```

## 各ディレクトリの責務

### `src/data`

データセット表現と CV 分割を担当します。

- `dataset.py`
  - `Dataset` を定義
  - CSV からの読み込み
  - 複数 `Dataset` の連結
- `group_cv.py`
  - グループ単位の CV 分割を提供

### `src/preprocess`

前処理の共通インターフェース、スペクトル変換、特徴量抽出を担当します。

- `core/`
  - `base.py`
    - `BasePreprocessor`, `ComposablePreprocessor`
  - `pipeline.py`
    - `PreprocessingPipeline`
  - `feature_union.py`
    - 複数 branch の特徴量横結合
  - `preprocessor_builder.py`
    - `preprocess_params.json` から前処理を構築
- `transforms/`
  - `identity.py`
  - `snv.py`
  - `savgol.py`
- `features/`
  - `interval_features.py`
  - `dwt_features.py`
  - `water_band_summary.py`
  - `group_sequence_features.py`

### `src/model`

モデル本体、学習器、評価、可視化、MLflow 連携を担当します。

- `base_model.py`
  - モデル共通インターフェース
- `model_builder.py`
  - `params.json` からモデルを構築
- `regressors/`
  - 具体的なモデル実装
- `trainers/`
  - CV 学習、全件学習のコアロジック
- `eval.py`
  - OOF 評価
- `plot.py`
  - Plotly による可視化
- `mlflow_handler.py`
  - モデルの保存 / 読み込み

### `src/recipes`

前処理名とモデル名をまとめた recipe 定義を担当します。

- `base.py`
  - `Recipe`
- `recipes.py`
  - 既定 recipe の定義
- `builder.py`
  - recipe 名から recipe を取得

### `src/pipeline`

Step と State を用いた実行フレームワークを担当します。

- `pipeline.py`
  - `BaseStep`, `Pipeline`, `BranchStep`
- `state.py`
  - `TrainState`, `PredictState`
- `steps/`
  - 学習、評価、予測、MLflow 実行などの各 Step
- `factory.py`
  - CV パイプラインと提出用パイプラインの組み立て

### `src/cli.py`

CLI エントリポイントです。

- `cv`
  - CV 学習と評価を実行
- `submission`
  - 全件学習して test データを予測

### `tests`

`src` の構成に合わせて、責務ごとにテストを分割しています。  
各モジュールの単体テストと、パイプラインの end-to-end テストを置いています。

## 中核オブジェクト

### `Dataset`

学習・前処理・推論で共通に使うデータコンテナです。

- `X`
- `y`
- `groups`
- `sample_id`
- `wavenumbers`

このプロジェクトでは、DataFrame を各所に持ち回すのではなく、まず `Dataset` に変換して扱います。

### `Recipe`

前処理名とモデル名をまとめた実行単位です。

- `name`
- `preprocessor_name`
- `model_name`

前処理本体の構成は `preprocess_params.json`、モデル本体の構成は `params.json` にあり、`Recipe` はその参照名だけを持ちます。

### `State`

Step 間で受け渡す共有状態です。

主な属性:

- `dataset`
- `train_dataset`
- `test_dataset`
- `preprocessor`
- `model`
- `oof_predictions`
- `test_predictions`
- `metrics`
- `run_id`

## 全体フロー

### 1. CV 学習フロー

```text
train.csv
  -> Dataset.from_csv()
  -> TrainState(dataset=...)
  -> build_cv_pipeline()
  -> TrainCV
  -> EvaluateOOF
  -> PlotOOF
  -> metrics / plots / MLflow run
```

内部では `TrainCV` が以下を行います。

- `RecipeBuilder` から recipe を取得
- fold ごとに `PreprocessorBuilder` で前処理器を作る
- `ModelBuilder` でモデルを生成
- OOF 予測を作成して `state.oof_predictions` に保存
- OOF 作成後、同じ recipe で全件再学習したモデルを `state.model` に保存

### 2. 提出用予測フロー

```text
train.csv + test.csv
  -> Dataset.from_csv()
  -> TrainState(train_dataset=..., test_dataset=...)
  -> build_submission_pipeline()
  -> TrainFull
  -> PredictTest
  -> submission.csv
```

内部では `TrainFull` が全学習データで前処理とモデルを学習し、`PredictTest` が test データへ適用します。

## モジュール間の関係

大まかな依存関係は次の通りです。

```text
CLI
  -> pipeline.factory
    -> pipeline.steps
      -> model.trainers
      -> model.eval
      -> model.plot
    -> recipes.builder
      -> recipes.recipes
    -> preprocess.core.preprocessor_builder
      -> preprocess.transforms
      -> preprocess.features
    -> model.model_builder
      -> model.regressors
  -> data.dataset
```

重要なのは、役割ごとに依存方向をなるべく一方向に保っている点です。

- `Recipe` はモデル実行の条件を定義するが、実行自体はしない
- `Trainer` は学習ロジックを持つが、CLI は知らない
- `Step` は orchestration に集中し、詳細ロジックは `Trainer` や `Evaluator` に委譲する

## 拡張ポイント

今後機能を増やすときの主な拡張ポイントは次の通りです。

### 新しい前処理を追加したい場合

- `src/preprocess/transforms/` または `src/preprocess/features/` に前処理クラスを追加
- `src/preprocess/core/preprocessor_builder.py` の build 分岐を追加
- `preprocess_params.json` から参照できるようにする

### 新しいモデルを追加したい場合

- `src/model/regressors/` などにモデル実装を追加
- `ModelBuilder` の build 分岐を追加
- 必要に応じて `MLflowHandler` の flavor 対応を追加

### 新しい recipe を追加したい場合

- `src/recipes/recipes.py` に recipe を追加
- `Recipe.preprocessor_name` が参照する前処理を `preprocess_params.json` に追加
- `Recipe.model_name` が参照するモデルを `params.json` に追加

### 新しい Step を追加したい場合

- `src/pipeline/steps/` に Step を追加
- 必要なら `factory.py` で既存パイプラインへ組み込む

## 関連ドキュメント

詳細は以下を参照してください。

- `docs/adding_features_preprocess_recipes.md` - 特徴量・前処理・レシピの追加手順
- `docs/design/recipe.md`
- `docs/design/preprocess.md`
- `docs/design/pipeline.md`
- `docs/design/composable_learning_pipeline.md` - X 前処理・y 変換・データ拡張を統合する設計方針
- `docs/data_info.md`
- `docs/knowledge.md`
