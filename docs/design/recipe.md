# Recipe ガイド

## 概要

`Recipe` は、前処理とモデル条件をひとまとめにした実行単位です。  
CV 学習と提出用学習で同じ条件を再利用するために使います。

このプロジェクトでは `src/recipes/base.py` で次のように定義されています。

```python
@dataclass(frozen=True)
class Recipe:
    name: str
    preprocessor_factory: Callable[[], BasePreprocessor]
    model_name: str
```

持っている情報はシンプルです。

- `name`: recipe 名
- `preprocessor_factory`: 前処理パイプラインを毎回新しく作る関数
- `model_name`: `ModelBuilder` に渡すモデル名

## なぜ factory にしているか

CV では fold ごとに新しい前処理器を作る必要があります。  
前処理器インスタンスそのものを `Recipe` に持たせると、学習済み状態が fold 間で共有される危険があります。

そのため、`preprocessor_factory` で毎回新しい前処理パイプラインを返す形にしています。

## 定義場所

既定の recipe は `src/recipes/default_recipes.py` に定義します。

```python
from src.preprocess.transforms.identity import IdentityPreprocessor
from src.preprocess.core.pipeline import PreprocessingPipeline
from src.recipes.base import Recipe


def make_identity_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline() >> IdentityPreprocessor()


RECIPES = {
    "base_pls": Recipe(
        name="base_pls",
        preprocessor_factory=make_identity_pipeline,
        model_name="base_pls",
    ),
}
```

## 取得方法

`RecipeBuilder` を使って名前から取得します。

```python
from src.recipes.builder import RecipeBuilder

recipe = RecipeBuilder().build("base_pls")
```

## パイプラインでの使われ方

CV 学習では `TrainCV` 内で recipe が使われます。

- fold ごとに `recipe.preprocessor_factory()` を呼ぶ
- `recipe.model_name` を使って `ModelBuilder` からモデルを作る

提出用学習でも同じ recipe を使うため、CV と提出で条件のずれを防げます。

## 新しい recipe の追加

新しい recipe を追加する手順は次の通りです。

1. 必要な前処理パイプラインを関数で定義する
2. `RECIPES` に `Recipe(...)` を追加する
3. `model_name` には `params.json` に存在するモデル名を指定する

例:

```python
def make_my_pipeline() -> PreprocessingPipeline:
    return PreprocessingPipeline() >> IdentityPreprocessor()


RECIPES["my_pls"] = Recipe(
    name="my_pls",
    preprocessor_factory=make_my_pipeline,
    model_name="base_pls",
)
```

## 設計上の意図

この設計は、次の点を重視しています。

- JSON に前処理の複雑な構造を書かなくてよい
- Python なので型補完や refactor が効く
- CV と提出用で同じ recipe をそのまま使える
- 前処理やモデル構成を増やしても責務が分かれたまま保てる
