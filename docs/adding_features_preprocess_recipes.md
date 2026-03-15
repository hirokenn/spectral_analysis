# 特徴量・前処理・レシピの追加方法

このドキュメントでは、特徴量、前処理、レシピを追加する手順をまとめます。  
既存の設計（`docs/design/preprocess.md`, `docs/design/recipe.md`）を前提に、実務的な手順に焦点を当てています。

---

## 1. 特徴量の追加

特徴量抽出器は、スペクトルやメタデータから新しい特徴量を生成する処理です。  
`Dataset` を入力として受け取り、`ds.with_features(...)` で特徴量を付加した `Dataset` を返します。

### 1.1 実装の配置

- **配置場所**: `src/preprocess/features/`
- **既存例**: `interval_features.py`, `dwt_features.py`, `water_band_summary.py`, `group_sequence_features.py`, `pls_oof_feature.py`

### 1.2 実装の要件

`BasePreprocessor` プロトコルに従い、以下を実装します。

- `fit(ds: Dataset) -> None`: 学習時パラメータ（区間インデックスなど）を保持
- `transform(ds: Dataset) -> Dataset`: 特徴量を計算し `ds.with_features(features, feature_names=...)` で返す
- `fit_transform(ds: Dataset) -> Dataset`: `fit` 後に `transform` を実行

`ComposablePreprocessor` を継承すると、`>>` でパイプラインに組み込めます。

```python
# src/preprocess/features/interval_features.py の例
@dataclass
class IntervalMeanFeatureExtractor(ComposablePreprocessor):
    interval_size: int = 100
    interval_indices_: list[np.ndarray] = field(default_factory=list, init=False)
    # ...

    def fit(self, ds: Dataset) -> None:
        self.interval_indices_ = _build_interval_indices(ds.wavenumbers, self.interval_size)
        # ...

    def transform(self, ds: Dataset) -> Dataset:
        features = np.stack([np.mean(ds.X[:, indices], axis=1) for indices in self.interval_indices_], axis=1)
        return ds.with_features(features, feature_names=self.feature_names_)
```

### 1.3 追加手順

| ステップ | 作業内容 |
|----------|----------|
| 1 | `src/preprocess/features/` に新規モジュールを作成し、特徴量クラスを実装する |
| 2 | `src/preprocess/features/__init__.py` にクラスを export する |
| 3 | `src/preprocess/core/preprocessor_builder.py` に `BuildType` を追加する |
| 4 | `PreprocessorBuilder._build_single_preprocessor` に分岐を追加する |
| 5 | `preprocess_params.json` に設定ノードを追加する |

### 1.4 PreprocessorBuilder への登録例

```python
# preprocessor_builder.py の BuildType に追加
class BuildType(StrEnum):
    # ...
    MY_FEATURE = "my_feature"

# _build_single_preprocessor に分岐を追加
from src.preprocess.features.my_feature import MyFeatureExtractor

if build_type == BuildType.MY_FEATURE:
    return MyFeatureExtractor(**params)
```

### 1.5 preprocess_params.json の定義例

```json
{
    "my_feature_default": {
        "build_type": "my_feature",
        "params": {
            "param_a": 10,
            "param_b": "value"
        }
    }
}
```

共通パラメータを共有する場合は `extends` を使います。

```json
{
    "_shared_my_feature": {
        "params": { "param_a": 10 }
    },
    "my_feature_default": {
        "extends": "_shared_my_feature",
        "build_type": "my_feature",
        "params": { "param_b": "value" }
    }
}
```

---

## 2. 前処理（変換）の追加

変換は、スペクトル全体に直接作用する処理です。  
SNV、Savitzky-Golay など、行単位の正規化や平滑化が該当します。

### 2.1 実装の配置

- **配置場所**: `src/preprocess/transforms/`
- **既存例**: `identity.py`, `snv.py`, `savgol.py`

### 2.2 実装の要件

特徴量と同様に `BasePreprocessor` に従います。  
変換後もスペクトル形式を保つ場合は `ds.with_features(transformed)` で返します。

```python
# src/preprocess/transforms/snv.py の例
@dataclass
class SNVPreprocessor(ComposablePreprocessor):
    def fit(self, ds: Dataset) -> None:
        pass  # 学習パラメータなし

    def transform(self, ds: Dataset) -> Dataset:
        mean = np.mean(ds.X, axis=1, keepdims=True)
        std = np.std(ds.X, axis=1, keepdims=True)
        transformed = (ds.X - mean) / np.where(std == 0.0, 1.0, std)
        return ds.with_features(transformed)
```

### 2.3 追加手順

| ステップ | 作業内容 |
|----------|----------|
| 1 | `src/preprocess/transforms/` に新規モジュールを作成し、変換クラスを実装する |
| 2 | `src/preprocess/core/preprocessor_builder.py` に `BuildType` を追加する |
| 3 | `PreprocessorBuilder._build_single_preprocessor` に分岐を追加する |
| 4 | `preprocess_params.json` に設定ノードを追加する |

---

## 3. パイプライン・FeatureUnion の組み立て

単体の特徴量や変換を、`pipeline` と `feature_union` で組み合わせます。

### 3.1 pipeline（直列連結）

前処理を順番に適用します。

```json
{
    "my_pipeline": {
        "build_type": "pipeline",
        "steps": ["snv", "interval_mean_100", "water_band_summary_default"]
    }
}
```

### 3.2 feature_union（並列結合）

複数 branch の出力を横結合します。

```json
{
    "my_spectral_union": {
        "build_type": "feature_union",
        "branches": [
            "interval_mean_100",
            "interval_slope_100",
            "dwt_db4_level3",
            "water_band_summary_default"
        ]
    }
}
```

### 3.3 組み合わせ例

変換 → 特徴量セット → 別特徴量 を結合する構成:

```json
{
    "snv_my_branch": {
        "build_type": "pipeline",
        "steps": ["snv", "my_spectral_union"]
    },
    "my_full_pipeline": {
        "build_type": "feature_union",
        "branches": [
            "snv_my_branch",
            "group_sequence_features_default"
        ]
    }
}
```

---

## 4. レシピの追加

レシピは、前処理名とモデル名をまとめた実行単位です。  
CV 学習と提出用学習で同じ条件を再利用するために使います。

### 4.1 Recipe の定義

```python
@dataclass(frozen=True)
class Recipe:
    name: str
    preprocessor_name: str
    model_name: str
    target_transform_name: str = "identity"
```

### 4.2 追加手順

| ステップ | 作業内容 |
|----------|----------|
| 1 | `preprocess_params.json` に前処理パイプラインを定義する（未定義の場合） |
| 2 | `params.json` にモデルを定義する（未定義の場合） |
| 3 | `src/recipes/recipes.py` の `_build_default_recipes` にレシピを追加する |

### 4.3 recipes.py への追加例

```python
# 通常レシピ（前処理名とモデル名が同じ場合）
_register_standard_recipe(
    recipes,
    name="my_recipe",
    preprocessor_name="my_full_pipeline",
)

# 前処理とモデル名が異なる場合
recipes["my_recipe"] = _make_recipe(
    "my_recipe",
    preprocessor_name="my_full_pipeline",
    model_name="snv_lgbm",
)

# ターゲット変換を変える場合（例: log1p）
recipes["my_recipe_log1p"] = _make_recipe(
    "my_recipe_log1p",
    preprocessor_name="my_full_pipeline",
    model_name="snv_lgbm",
    target_transform_name="log1p",
)
```

### 4.4 実行方法

```bash
# CV 学習・評価
spectral-analysis cv --recipe-name my_recipe --experiment-name spectral_analysis

# 提出用（全件学習 → test 予測）
spectral-analysis submission --recipe-name my_recipe --train-path data/train.csv --test-path data/test.csv
```

---

## 5. チェックリスト

新しい特徴量・前処理・レシピを追加した際の確認項目です。

### 特徴量・前処理

- [ ] `fit` / `transform` / `fit_transform` が正しく実装されている
- [ ] `ds.with_features(...)` で適切な `feature_names` を渡している
- [ ] `PreprocessorBuilder` に `BuildType` と分岐を追加した
- [ ] `preprocess_params.json` に設定を追加した
- [ ] 循環参照がない（`extends` や `branches` の参照チェック）

### レシピ

- [ ] `preprocessor_name` が `preprocess_params.json` に存在する
- [ ] `model_name` が `params.json` に存在する
- [ ] `recipes.py` にレシピを登録した
- [ ] `spectral-analysis cv --recipe-name <name>` で実行できる

### 品質

- [ ] `uv run ruff check src --fix`
- [ ] `uv run black .`
- [ ] `uv run mypy src`
- [ ] `uv run pytest`

---

## 6. 参照

- `docs/design/preprocess.md` - 前処理の設計とインターフェース
- `docs/design/recipe.md` - レシピの設計
- `docs/architecture.md` - 全体アーキテクチャ
- `preprocess_params.json` - 前処理設定の実例
- `params.json` - モデル設定の実例
