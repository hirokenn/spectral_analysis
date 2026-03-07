# spectral_analysis

## プロジェクト概要

木材の近赤外（NIR）スペクトルから含水率を予測する回帰タスクを扱うプロジェクトです。  
ケモメトリクス的アプローチと機械学習アプローチを比較しながら、スペクトルデータの前処理・学習・評価を進めることを目的としています。

## データ情報ドキュメント

データセットの基本情報（サイズ、カラム定義、統計量）は以下にまとまっています。

- `docs/data_info.md`

## 環境構築

### 前提

- Python 3.12 以上
- `uv` がインストール済みであること

### セットアップ手順

1. 仮想環境を作成

```bash
uv venv
```

2. 仮想環境を有効化

```bash
source .venv/bin/activate
```

3. 依存関係をインストール

```bash
uv sync --all-groups
```

4. コマンド実行確認（任意）

```bash
uv run python --version
```

### 品質チェックコマンド

```bash
uv run mypy src
uv run black .
uv run ruff check src
uv run pytest
```
