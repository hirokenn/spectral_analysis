from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.data.dataset import Dataset
from src.model.model_builder import ModelBuilder
from src.pipeline import build_cv_pipeline, build_submission_pipeline
from src.pipeline.state import TrainState
from src.preprocess.core.preprocessor_builder import PreprocessorBuilder
from src.recipes.default_recipes import RECIPES
from src.recipes.builder import RecipeBuilder

DEFAULT_TRAIN_PATH = "data/train.csv"
DEFAULT_TEST_PATH = "data/test.csv"
DEFAULT_SUBMISSION_DIR = "submission"


def build_parser() -> argparse.ArgumentParser:
    """CLI パーサを構築する。"""
    parser = argparse.ArgumentParser(description="spectral_analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cv_parser = subparsers.add_parser("cv", help="CV 学習と評価を実行する")
    _add_common_args(cv_parser)
    cv_parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN_PATH,
        help="学習用 CSV パス",
    )
    cv_parser.set_defaults(func=run_cv)

    submission_parser = subparsers.add_parser(
        "submission",
        help="全件学習して test を予測する",
    )
    _add_common_args(submission_parser)
    submission_parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN_PATH,
        help="学習用 CSV パス",
    )
    submission_parser.add_argument(
        "--test-path",
        default=DEFAULT_TEST_PATH,
        help="テスト用 CSV パス",
    )
    submission_parser.add_argument(
        "--output-path",
        default=None,
        help="提出 CSV の出力先",
    )
    submission_parser.set_defaults(func=run_submission)

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """CV / submission 共通引数を追加する。"""
    parser.add_argument(
        "--recipe-name",
        default="base_pls",
        choices=sorted(RECIPES.keys()),
        help="実行する recipe 名",
    )
    parser.add_argument(
        "--experiment-name",
        default="spectral_analysis",
        help="MLflow experiment 名",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run 名")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="パイプライン実行ログを表示する（既定: 有効）",
    )


def run_cv(args: argparse.Namespace) -> int:
    """CV パイプラインを実行する。"""
    train_dataset = Dataset.from_csv(args.train_path)
    state = TrainState(dataset=train_dataset)
    run_name = args.run_name if args.run_name is not None else args.recipe_name
    pipeline = build_cv_pipeline(
        recipe_builder=RecipeBuilder(),
        model_builder=ModelBuilder(),
        preprocessor_builder=PreprocessorBuilder(),
        recipe_name=args.recipe_name,
        run_name=run_name,
        experiment_name=args.experiment_name,
        verbose=args.verbose,
    )
    updated = pipeline.run(state)

    print(
        json.dumps(
            {
                "run_id": updated.run_id,
                "overall_rmse": updated.metrics.get("overall_rmse"),
                "group_rmse_mean": updated.metrics.get("group_rmse_mean"),
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_submission(args: argparse.Namespace) -> int:
    """提出用パイプラインを実行して予測 CSV を保存する。"""
    train_dataset = Dataset.from_csv(args.train_path)
    test_dataset = Dataset.from_csv(args.test_path)
    state = TrainState(train_dataset=train_dataset, test_dataset=test_dataset)
    pipeline = build_submission_pipeline(
        recipe_builder=RecipeBuilder(),
        model_builder=ModelBuilder(),
        preprocessor_builder=PreprocessorBuilder(),
        recipe_name=args.recipe_name,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        verbose=args.verbose,
    )
    updated = pipeline.run(state)
    assert updated.test_predictions is not None

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else Path(DEFAULT_SUBMISSION_DIR) / f"{args.recipe_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for sample_id, pred in zip(test_dataset.sample_id, updated.test_predictions):
            writer.writerow([int(sample_id), float(pred)])

    print(
        json.dumps(
            {
                "run_id": updated.run_id,
                "output_path": str(output_path),
                "n_predictions": int(updated.test_predictions.shape[0]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI のエントリポイント。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
