class PipelineError(Exception):
    """パイプライン実行時の基底例外"""

    pass


class InvalidStateError(PipelineError):
    """Step 実行時の state 形状が不正な場合の例外。"""

    pass
