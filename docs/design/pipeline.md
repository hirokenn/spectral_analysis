# Pipeline ガイド

## 基本的な考え方

このプロジェクトでは、処理を小さな `Step` に分け、それらを `>>` で連結して `Pipeline` を作ります。  
各 Step は `State` を受け取り、必要な情報を更新して次の Step に渡します。

- `BaseStep`: 各処理単位
- `State`: Step 間で受け渡す状態
- `Pipeline`: Step を順番に実行するコンテナ

## Step の作り方

Step は `BaseStep` を継承し、`execute()` を実装します。

```python
from typing import Any

from src.pipeline.pipeline import BaseStep
from src.pipeline.state import StateLike, TrainState


class MyStep(BaseStep):
    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state, allowed_types=(TrainState,))
        return state
```

`_require_state()` を使うと、Step の入口で型や必須属性を検証できます。

## パイプラインの組み立て

パイプラインは `>>` で結合します。

```python
from src.pipeline.state import TrainState
from src.pipeline.steps.run_step import MlflowEndRun, MlflowStartRun

pipeline = MlflowStartRun() >> MyStep() >> MlflowEndRun()

state = TrainState()
result = pipeline.run(state)
```

## このプロジェクトの主要パイプライン

### CV パイプライン

CV 用パイプラインは `src.pipeline.factory.build_cv_pipeline()` で構築します。

```python
from src.model.model_builder import ModelBuilder
from src.pipeline import build_cv_pipeline
from src.pipeline.state import TrainState
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder
from src.recipes.builder import RecipeBuilder

pipeline = build_cv_pipeline(
    recipe_builder=RecipeBuilder(),
    model_builder=ModelBuilder(),
    preprocessor_builder=PreprocessorBuilder(),
    recipe_name="snv_lgbm",
)

state = TrainState(dataset=train_ds)
result = pipeline.run(state)
```

実行される Step は次の通りです。

- `MlflowStartRun`
- `TrainCV`
- `EvaluateOOF`
- `PlotOOF` (`plot_oof=True` のとき)
- `MlflowEndRun`

`TrainCV` の内部では次を行います。

- `RecipeBuilder` で recipe を取得
- `PreprocessorBuilder` で前処理を構築
- `ModelBuilder` でモデルを構築
- `LOSOCV` で fold を回して OOF を作成
- OOF 作成後、同じ recipe で全件再学習して `state.model` に保持

### 提出用パイプライン

提出用パイプラインは `src.pipeline.factory.build_submission_pipeline()` で構築します。

```python
from src.model.model_builder import ModelBuilder
from src.pipeline import build_submission_pipeline
from src.pipeline.state import TrainState
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder
from src.recipes.builder import RecipeBuilder

pipeline = build_submission_pipeline(
    recipe_builder=RecipeBuilder(),
    model_builder=ModelBuilder(),
    preprocessor_builder=PreprocessorBuilder(),
    recipe_name="snv_lgbm",
)

state = TrainState(train_dataset=train_ds, test_dataset=test_ds)
result = pipeline.run(state)
pred = result.test_predictions
```

実行される Step は次の通りです。

- `MlflowStartRun`
- `TrainFull`
- `PredictTest`
- `MlflowEndRun`

## 現在の実装上の注意

- `LOSOCV` は既定で各グループを 1 回ずつ validation に使います。
- `LOSOCV` は `tqdm` による進捗表示を持ちます。
- `build_cv_pipeline()` は既定で `PlotOOF()` を含むため、Plotly HTML の artifact も作成します。
- `Pipeline.verbose=True` のとき、各 Step の実行時間とログが標準出力に表示されます。

## 実行ログの例

```text
パイプラインを実行中...
--------------------------------------------------------------------------------
[1] MlflowStartRun: 0.579s - run_id=...
LOSOCV: 100%|██████████| 19/19 [00:26<00:00,  1.40s/split]
[2] TrainCV: 26.801s - recipe=snv_lgbm, splits=19, covered=1322/1322
[3] EvaluateOOF: 0.003s - overall_rmse=..., group_rmse_mean=...
--------------------------------------------------------------------------------
総実行時間: ...
```
