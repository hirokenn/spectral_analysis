from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple

from src.exceptions import InvalidStateError
from src.pipeline.state import PredictState, StateLike, TrainState


class BaseStep(ABC):
    """
    すべての Step が継承する抽象基底クラス。
    - execute() は必ず override する（挙動の入口）
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.log_out: Optional[str] = None

    @abstractmethod
    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        """Step のエントリポイント"""
        pass

    def __call__(self, state: StateLike, **kwargs: Any) -> StateLike:
        return self.execute(state, **kwargs)

    def __rshift__(self, other: "BaseStep | Pipeline") -> "Pipeline":
        if isinstance(other, Pipeline):
            return Pipeline([self, *other.steps])
        elif isinstance(other, BaseStep):
            return Pipeline([self, other])
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    def _require_state(
        self,
        state: Any,
        *,
        allowed_types: Optional[Tuple[type, ...]] = (TrainState, PredictState),
        require_attrs: Iterable[str] = (),
        require_non_none: Iterable[str] = (),
    ) -> None:
        """
        Step の入口で state 形状を検証する。
        - state が None の場合は即エラー
        - allowed_types が指定されていて型が一致しない場合はエラー
        - require_attrs の属性が無い場合はエラー
        - require_non_none の属性が None の場合はエラー
        """
        step_name = self.__class__.__name__

        if state is None:
            raise InvalidStateError(f"{step_name}: state が None です")

        if allowed_types is not None and not isinstance(state, allowed_types):
            raise TypeError(
                f"{step_name}: state の型が想定外です "
                f"(expected={allowed_types}, actual={type(state)})"
            )

        for attr in require_attrs:
            if not hasattr(state, attr):
                raise TypeError(f"{step_name}: state に '{attr}' 属性がありません")

        for attr in require_non_none:
            if getattr(state, attr, None) is None:
                raise InvalidStateError(f"{step_name}: state.{attr} が None です")


class BranchStep(BaseStep):
    def __init__(
        self,
        condition_fn: Callable[[StateLike], bool],
        true_steps: Optional[BaseStep | Pipeline] = None,
        false_steps: Optional[BaseStep | Pipeline] = None,
    ):
        super().__init__()
        self.condition_fn = condition_fn
        self.true_steps = true_steps
        self.false_steps = false_steps

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state)
        steps = self.true_steps if self.condition_fn(state) else self.false_steps

        if steps is None:
            return state
        elif isinstance(steps, Pipeline):
            return steps.run(state)
        else:
            return steps(state)


class Pipeline:
    def __init__(self, steps: List[BaseStep], verbose: bool = True):
        self.steps = steps
        self.verbose = verbose
        self.step_counter = 0

    def __iter__(self) -> Iterator[BaseStep]:
        return iter(self.steps)

    def __rshift__(self, other: BaseStep | Pipeline) -> Pipeline:
        if isinstance(other, BaseStep):
            return Pipeline(self.steps + [other], verbose=self.verbose)
        elif isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps, verbose=self.verbose)
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

    def _execute_step_with_logging(
        self,
        step: BaseStep,
        state: StateLike,
        step_index: int,
        parent_step: Optional[str] = None,
    ) -> StateLike:
        """ステップを実行し、実行状況をログ出力"""

        step_name = step.__class__.__name__
        step_start = time.time()

        try:
            if isinstance(step, BranchStep):
                condition_result = step.condition_fn(state)
                steps_to_execute = (
                    step.true_steps if condition_result else step.false_steps
                )

                branch_start = time.time()

                if self.verbose:
                    print(
                        f"[{step_index + 1}] {step_name} ({'True' if condition_result else 'False'})"
                    )

                if steps_to_execute is None:
                    pass
                elif isinstance(steps_to_execute, Pipeline):
                    for _sub_idx, sub_step in enumerate(steps_to_execute.steps):
                        state = self._execute_step_with_logging(
                            sub_step, state, self.step_counter, parent_step=step_name
                        )
                        self.step_counter += 1
                elif isinstance(steps_to_execute, BaseStep):
                    state = self._execute_step_with_logging(
                        steps_to_execute,
                        state,
                        self.step_counter,
                        parent_step=step_name,
                    )
                    self.step_counter += 1

                branch_end = time.time()
                branch_time = branch_end - branch_start

                if self.verbose:
                    log_info = f" - {step.log_out}" if step.log_out is not None else ""
                    print(
                        f"  [{step_index + 1}] {step_name}: {branch_time:.3f}s{log_info}"
                    )

            else:
                result_state = step(state)
                state = result_state  # type: ignore
                step_end = time.time()
                execution_time = step_end - step_start

                if self.verbose:
                    indent = "  " if parent_step else ""
                    log_info = f" - {step.log_out}" if step.log_out is not None else ""
                    print(
                        f"{indent}[{step_index + 1}] {step_name}: {execution_time:.3f}s{log_info}"
                    )

            return state

        except Exception as e:
            step_end = time.time()
            execution_time = step_end - step_start

            error_msg = str(e)
            if self.verbose:
                indent = "  " if parent_step else ""
                print(
                    f"{indent}[{step_index + 1}] {step_name}: {execution_time:.3f}s (エラー: {error_msg})"
                )
            raise

    def run(self, state: StateLike) -> StateLike:
        """パイプラインを実行"""
        if self.verbose:
            print("パイプラインを実行中...")
            print("-" * 80)

        self.step_counter = 0
        start_time = time.time()

        for idx, step in enumerate(self.steps):
            try:
                state = self._execute_step_with_logging(step, state, idx)
                self.step_counter += 1
            except Exception as e:
                if self.verbose:
                    print(f"パイプライン実行中にエラーが発生しました: {e}")
                raise

        if self.verbose:
            total_time = time.time() - start_time
            print("-" * 80)
            print(f"総実行時間: {total_time:.3f}s")

        return state
