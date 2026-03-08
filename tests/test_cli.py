from __future__ import annotations

import csv
import json
from pathlib import Path

from src.cli import main


def write_params_json(path: Path) -> None:
    """CLI テスト用の params.json を作成する。"""
    path.write_text(
        json.dumps(
            {
                "base_pls": {
                    "build_type": "pls",
                    "params": {"n_components": 1, "scale": True},
                }
            }
        ),
        encoding="utf-8",
    )


def write_train_csv(path: Path) -> None:
    """CLI テスト用の train.csv を作成する。"""
    with path.open("w", encoding="cp932", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample number", "species number", "4000", "5000", "含水率"])
        writer.writerow([1, 1, 0.1, 0.2, 1.0])
        writer.writerow([2, 2, 0.2, 0.3, 2.0])
        writer.writerow([3, 3, 0.3, 0.4, 3.0])
        writer.writerow([4, 4, 0.4, 0.5, 4.0])


def write_test_csv(path: Path) -> None:
    """CLI テスト用の test.csv を作成する。"""
    with path.open("w", encoding="cp932", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample number", "species number", "4000", "5000"])
        writer.writerow([10, 5, 1.0, 1.1])
        writer.writerow([11, 5, 1.2, 1.3])


def test_cli_cv_runs(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    train_path = tmp_path / "train.csv"
    write_train_csv(train_path)

    exit_code = main(
        [
            "cv",
            "--train-path",
            str(train_path),
            "--experiment-name",
            "test_cli_cv",
            "--no-verbose",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code == 0
    assert payload["run_id"] is not None
    assert "overall_rmse" in payload
    assert "group_rmse_mean" in payload


def test_cli_submission_runs_and_writes_csv(
    tmp_path: Path, capsys, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "submission" / "base_pls.csv"
    write_train_csv(train_path)
    write_test_csv(test_path)
    write_params_json(tmp_path / "params.json")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "submission",
            "--train-path",
            str(train_path),
            "--test-path",
            str(test_path),
            "--experiment-name",
            "test_cli_submission",
            "--no-verbose",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert exit_code == 0
    assert payload["run_id"] is not None
    assert Path(payload["output_path"]).resolve() == output_path.resolve()
    assert output_path.exists()

    rows = list(csv.reader(output_path.open("r", encoding="utf-8", newline="")))
    assert rows[0] == ["sample number", "含水率"]
    assert len(rows) == 3
