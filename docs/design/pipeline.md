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

`Pipeline([...])` でも作れますが、このプロジェクトでは可読性のため `>>` を基本とします。

## このプロジェクトの主要パイプライン

### CV パイプライン

CV 用パイプラインは `src.pipeline.factory.build_cv_pipeline()` で構築します。

```python
from src.model.model_builder import ModelBuilder
from src.pipeline import build_cv_pipeline
from src.pipeline.state import TrainState
from src.recipes.builder import RecipeBuilder

pipeline = build_cv_pipeline(
    recipe_builder=RecipeBuilder(),
    model_builder=ModelBuilder(),
    recipe_name="base_pls",
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

このパイプラインは以下を行います。

- recipe に従って各 fold で前処理とモデルを再構築
- OOF 予測を作成
- RMSE を `state.metrics` に保存
- 相関プロットや残差ヒストグラムを MLflow に HTML 保存

### 提出用パイプライン

提出用パイプラインは `src.pipeline.factory.build_submission_pipeline()` で構築します。

```python
from src.model.model_builder import ModelBuilder
from src.pipeline import build_submission_pipeline
from src.pipeline.state import TrainState
from src.recipes.builder import RecipeBuilder

pipeline = build_submission_pipeline(
    recipe_builder=RecipeBuilder(),
    model_builder=ModelBuilder(),
    recipe_name="base_pls",
)

state = TrainState(train_dataset=train_ds, test_dataset=test_ds)
result = pipeline.run(state)
pred = result.test_predictions
```

実行される Step は次の通りです。

- `TrainFull`
- `PredictTest`

このパイプラインは以下を行います。

- 全学習データで前処理器とモデルを学習
- 学習済み前処理器を test データへ適用
- `state.test_predictions` に予測値を保存

## 実行ログ

`Pipeline.verbose` が `True` の場合、各 Step の実行時間とログが出力されます。

```python
pipeline = build_cv_pipeline(
    recipe_builder=RecipeBuilder(),
    model_builder=ModelBuilder(),
    recipe_name="base_pls",
    verbose=True,
)
```

出力例:

```text
パイプラインを実行中...
--------------------------------------------------------------------------------
[1] MlflowStartRun: 0.010s - run_id=...
[2] TrainCV: 0.254s - recipe=base_pls, splits=20, covered=1322/1322
[3] EvaluateOOF: 0.002s - overall_rmse=..., group_rmse_mean=...
--------------------------------------------------------------------------------
総実行時間: 0.310s
```