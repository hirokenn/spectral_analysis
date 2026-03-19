# Preprocess ガイド

## 概要

このプロジェクトの前処理は、`Dataset` を入力として受け取り、変換後の `Dataset` を返す小さな処理単位として実装します。  
構成は Python クラスで実装し、接続順や分岐は `preprocess_params.json` で定義します。

実装上は次の 3 層に分かれています。

- `src/preprocess/core/`: 抽象、合成、builder
- `src/preprocess/transforms/`: スペクトル全体へ直接作用する変換
- `src/preprocess/features/`: 新しい特徴量を生成する抽出器

## インターフェース

前処理は `src/preprocess/core/base.py` の `BasePreprocessor` に従います。

```python
@runtime_checkable
class BasePreprocessor(Protocol):
    def fit(self, ds: Dataset) -> None: ...
    def transform(self, ds: Dataset) -> Dataset: ...
    def fit_transform(self, ds: Dataset) -> Dataset: ...
```

最小限の契約だけに絞っているため、シンプルに実装できます。

`ComposablePreprocessor` を継承すると、`>>` を使って `PreprocessingPipeline` に組み込めます。

## PreprocessingPipeline

`PreprocessingPipeline` は複数前処理を順番に適用するクラスです。

```python
from src.preprocess.transforms.identity import IdentityPreprocessor
from src.preprocess.core.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline() >> IdentityPreprocessor()
```

複数つなぐ場合も同じです。

```python
pipeline = (
    PreprocessingPipeline()
    >> IdentityPreprocessor()
    >> IdentityPreprocessor()
)
```

`fit()`、`transform()`、`fit_transform()` は中の前処理を順に呼び出します。

## FeatureUnion

複数の branch で別々に特徴量を作り、最後に横結合したい場合は `FeatureUnion` を使います。

```python
from src.preprocess.core.feature_union import FeatureUnion
from src.preprocess.transforms.snv import SNVPreprocessor
from src.preprocess.features.interval_features import IntervalMeanFeatureExtractor
from src.preprocess.features.group_sequence_features import GroupSequenceFeatureExtractor

preprocessor = FeatureUnion(
    branches=[
        SNVPreprocessor() >> IntervalMeanFeatureExtractor(),
        GroupSequenceFeatureExtractor(),
    ]
)
```

`snv_lgbm` はこの形で、
- 片方の branch で `SNV` 後のスペクトル特徴
- もう片方の branch でグループ内順序特徴

を作って結合しています。

## 既存の前処理

### IdentityPreprocessor

`IdentityPreprocessor` は何もしない no-op 前処理です。  
recipe や pipeline の枠組みを保ったまま、前処理未適用の条件を表現するために使います。

```python
from src.preprocess.transforms.identity import IdentityPreprocessor

preprocessor = IdentityPreprocessor()
```

### 代表的な transforms

- `SNVPreprocessor`
  - 行ごとに平均 0、標準偏差 1 に正規化
- `SavitzkyGolayPreprocessor`
  - `scipy.signal.savgol_filter` による平滑化
- `IdentityPreprocessor`
  - no-op

### 代表的な feature extractors

- `IntervalMeanFeatureExtractor`
  - 固定点数ごとの区間平均
- `IntervalSlopeFeatureExtractor`
  - 固定点数ごとの一次回帰傾き
- `DWTFeatureExtractor`
  - 離散ウェーブレット係数の要約統計
- `WaterBandSummaryFeatureExtractor`
  - 水関連帯域の平均、標準偏差、面積など
- `GroupSequenceFeatureExtractor`
  - グループ内 `sample_id` 順序に基づく差分・移動統計

## recipe との関係

前処理は通常、recipe の `preprocessor_name` を起点に `PreprocessorBuilder` から生成します。

```python
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder

preprocessor = PreprocessorBuilder().build("snv_lgbm_pipeline")
```

こうしておくと、CV の fold ごとに新しい前処理器を安全に作れます。

## JSON 設定の考え方

`preprocess_params.json` では、前処理を名前付きノードとして定義します。

- 単体前処理
  - `build_type` と `params`
- 直列合成
  - `build_type: "pipeline"`
  - `steps`
- 並列合成
  - `build_type: "feature_union"`
  - `branches`

たとえば現在の `snv_lgbm` は概念的には次のような構成です。

```text
snv_lgbm_pipeline
  -> FeatureUnion
    -> snv_lgbm_snv_branch
      -> SNV
      -> spectral features
    -> group_sequence_features_default
```

## 新しい前処理の追加

新しい前処理を追加する場合は、`fit` / `transform` / `fit_transform` を実装します。

```python
from dataclasses import dataclass

from src.data.dataset import Dataset


@dataclass
class MyPreprocessor:
    def fit(self, ds: Dataset) -> None:
        _ = ds

    def transform(self, ds: Dataset) -> Dataset:
        return ds

    def fit_transform(self, ds: Dataset) -> Dataset:
        self.fit(ds)
        return self.transform(ds)
```

追加手順は次の通りです。

1. `src/preprocess/transforms/` または `src/preprocess/features/` にクラスを追加する
2. `src/preprocess/core/preprocessor_builder.py` に `build_type` を追加する
3. `preprocess_params.json` に設定ノードを追加する
4. 必要なら `src/recipes/recipes.py` のレシピからその前処理名を参照する

詳細な手順は `docs/adding_features_preprocess_recipes.md` を参照してください。

## 現在の主要前処理名

実装時によく参照する主な設定名は次の通りです。

- `base_identity_pipeline`
- `snv_pls_pipeline`
- `snv_lgbm_pipeline`
- `snv_lgbm_snv_branch`
- `snv_lgbm_spectral_union`
- `savgol_lgbm_pipeline`
- `group_sequence_features_default`

## 設計上の意図

この前処理設計は、次の点を重視しています。

- `Dataset` ベースで入出力を統一する
- 各前処理を小さく保つ
- 直列合成は `pipeline`、並列合成は `feature_union` に分離する
- Python 実装と JSON 定義を分け、再利用しやすくする
- CV と提出で同じ前処理定義を再利用する
