# Recipe ガイド

## 概要

`Recipe` は、前処理とモデル条件をひとまとめにした実行単位です。  
CV 学習と提出用学習で同じ条件を再利用するために使います。

このプロジェクトでは `src/recipes/base.py` で次のように定義されています。

```python
@dataclass(frozen=True)
class Recipe:
    name: str
    preprocessor_name: str
    model_name: str
```

持っている情報はシンプルです。

- `name`: recipe 名
- `preprocessor_name`: `PreprocessorBuilder` に渡す前処理設定名
- `model_name`: `ModelBuilder` に渡すモデル名

## なぜ名前参照にしているか

前処理の構造は `preprocess_params.json`、モデルの構造は `params.json` に分離しています。  
`Recipe` はその参照名だけを持つため、次の利点があります。

- `Recipe` 自体を小さく保てる
- 同じ前処理ノードを複数 recipe で共有できる
- CLI 側は recipe 名だけ知っていれば実行できる
- CV の fold ごとに `Builder` が毎回新しいインスタンスを作れる

## 定義場所

既定の recipe は `src/recipes/recipes.py` に定義します。

```python
from src.recipes.base import Recipe

RECIPES = {
    "base_pls": Recipe(
        name="base_pls",
        preprocessor_name="base_identity_pipeline",
        model_name="base_pls",
    )
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

- fold ごとに `recipe.preprocessor_name` を `PreprocessorBuilder` に渡す
- `recipe.model_name` を使って `ModelBuilder` からモデルを作る

提出用学習でも同じ recipe を使うため、CV と提出で条件のずれを防げます。

## 新しい recipe の追加

新しい recipe を追加する手順は次の通りです。

1. `preprocess_params.json` に前処理名を定義する
2. `params.json` にモデル名を定義する
3. `src/recipes/recipes.py` の `_build_default_recipes` 内でレシピを登録する

詳細な手順は `docs/adding_features_preprocess_recipes.md` を参照してください。

## 現在の recipe

現在の既定 recipe は次の通りです。

- `base_pls`
  - `preprocessor_name="base_identity_pipeline"`
  - `model_name="base_pls"`
- `snv_pls`
  - `preprocessor_name="snv_pls_pipeline"`
  - `model_name="snv_pls"`
- `snv_lgbm`
  - `preprocessor_name="snv_lgbm_pipeline"`
  - `model_name="snv_lgbm"`
- `savgol_lgbm`
  - `preprocessor_name="savgol_lgbm_pipeline"`
  - `model_name="savgol_lgbm"`

`snv_lgbm` は `FeatureUnion` を使って、
- `SNV` 後のスペクトル特徴
- グループ内順序由来の特徴

を結合する構成です。

## 設計上の意図

この設計は、次の点を重視しています。

- recipe 自体はシンプルな参照オブジェクトに保つ
- 前処理とモデルの詳細は JSON に分離して再利用しやすくする
- recipe 名を CLI / 実験管理の識別子として使いやすくする
- CV と提出用で同じ recipe をそのまま使える
- 前処理やモデル構成を増やしても責務が分かれたまま保てる
