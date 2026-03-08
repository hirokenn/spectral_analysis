# Preprocess ガイド

## 概要

前処理は `Dataset` を入力として受け取り、変換後の `Dataset` を返す小さな処理単位として扱います。  
複数の前処理は `PreprocessingPipeline` でまとめ、`>>` で連結します。

## インターフェース

前処理は `src/preprocess/base.py` の `BasePreprocessor` に従います。

```python
@runtime_checkable
class BasePreprocessor(Protocol):
    def fit(self, ds: Dataset) -> None: ...
    def transform(self, ds: Dataset) -> Dataset: ...
    def fit_transform(self, ds: Dataset) -> Dataset: ...
```

最小限の契約だけに絞っているため、シンプルに実装できます。

## PreprocessingPipeline

`PreprocessingPipeline` は複数前処理を順番に適用するクラスです。

```python
from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.pipeline import PreprocessingPipeline

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

## 既存の前処理

### IdentityPreprocessor

`IdentityPreprocessor` は何もしない no-op 前処理です。  
recipe や pipeline の枠組みを保ったまま、前処理未適用の条件を表現するために使います。

```python
from src.preprocess.identity import IdentityPreprocessor

preprocessor = IdentityPreprocessor()
```

## recipe との関係

前処理は通常、recipe の `preprocessor_factory` から生成します。

```python
def make_identity_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline() >> IdentityPreprocessor()
```

こうしておくと、CV の fold ごとに新しい前処理器を安全に作れます。

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

その後、`PreprocessingPipeline() >> MyPreprocessor()` の形で recipe に組み込みます。

## 設計上の意図

この前処理設計は、次の点を重視しています。

- `Dataset` ベースで入出力を統一する
- 各前処理を小さく保つ
- `>>` で処理順を読みやすく記述する
- CV と提出で同じ前処理定義を再利用する
