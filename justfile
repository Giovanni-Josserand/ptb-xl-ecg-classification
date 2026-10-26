default:
    @just --list

install:
    test -d .venv || python3 -m venv .venv
    uv sync

preprocess:
    uv run python src/ptb_xl_ecg_classification/preprocessing/create_clean_dataset.py

