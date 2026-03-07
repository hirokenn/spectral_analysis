# Pipeline ガイド

## 基本的なアイデア

Pipelineは、機械学習の処理を複数のステップに分割し、順番に実行するためのフレームワークです。

- **ステップ（Step）**: 各処理単位（データ読み込み、特徴量生成、モデル学習など）
- **状態（State）**: ステップ間でデータを受け渡すためのオブジェクト
- **パイプライン（Pipeline）**: 複数のステップを順番に実行するコンテナ

## 基本的な使い方

### 1. ステップの作成

ステップは `BaseStep` を継承し、`execute()` メソッドを実装します。

```python
from src.pipelines.pipeline import BaseStep
from src.pipelines.state import TrainState, StateLike

class MyStep(BaseStep):
    def execute(self, state: StateLike, **kwargs) -> StateLike:
        # stateを検証（オプション）
        self._require_state(state, require_attrs=['df_feature'])
        
        # 処理を実行
        # stateを更新
        state.df_feature = process_data(state.df_feature)
        
        return state
```

### 2. パイプラインの作成と実行

ステップを `>>` 演算子で連結してパイプラインを作成し、`run()` で実行します。

```python
from src.pipelines.pipeline import Pipeline
from src.pipelines.state import TrainState

# ステップを作成
step1 = LoadDataStep()
step2 = BuildFeatureStep()
step3 = TrainModelStep()

# パイプラインを作成
pipeline = step1 >> step2 >> step3

# または
pipeline = Pipeline([step1, step2, step3])

# 実行
initial_state = TrainState()
result_state = pipeline.run(initial_state, verbose=True)
```

### 3. 条件分岐

`BranchStep` を使用して条件に応じた分岐処理が可能です。

```python
from src.pipelines.pipeline import BranchStep

def should_use_cache(state: StateLike) -> bool:
    return state.feature_version is not None

# 条件分岐パイプライン
branch = BranchStep(
    condition_fn=should_use_cache,
    true_steps=LoadCacheStep(),  # キャッシュがある場合
    false_steps=BuildFeatureStep() >> SaveCacheStep()  # キャッシュがない場合
)

pipeline = LoadDataStep() >> branch >> TrainModelStep()
```

### 4. 状態（State）の検証

ステップ内で `_require_state()` を使用して、必要な属性が存在するか検証できます。

```python
def execute(self, state: StateLike, **kwargs) -> StateLike:
    # 特定の型のみ許可
    self._require_state(state, allowed_types=(TrainState,))
    
    # 特定の属性が存在することを確認
    self._require_state(state, require_attrs=['df_feature', 'df_label'])
    
    # 特定の属性がNoneでないことを確認
    self._require_state(state, require_non_none=['df_feature'])
    
    # 処理を実行
    return state
```

## 主なクラス

- **`Pipeline`**: ステップの実行コンテナ
- **`BaseStep`**: すべてのステップの基底クラス
- **`BranchStep`**: 条件分岐を行うステップ
- **`TrainState`**: 学習用の状態オブジェクト
- **`PredictState`**: 予測用の状態オブジェクト

## 実行ログ

`verbose=True` を指定すると、各ステップの実行時間とログが出力されます。

```python
pipeline.run(state, verbose=True)
```

出力例:
```
パイプラインを実行中...
--------------------------------------------------------------------------------
[1] LoadDataStep: 0.123s
[2] BuildFeatureStep: 0.456s
[3] TrainModelStep: 1.234s
--------------------------------------------------------------------------------
総実行時間: 1.813s
```