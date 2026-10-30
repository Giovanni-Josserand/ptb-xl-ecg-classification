default:
    @just --list

install:
    test -d .venv || python3 -m venv .venv
    uv sync

preprocess:
    uv run python src/ptb_xl_ecg_classification/preprocessing/create_clean_dataset.py

verify-preprocess:
    uv run python src/ptb_xl_ecg_classification/preprocessing/verify_dataset.py

random-forest:
    uv run python src/ptb_xl_ecg_classification/random_forest.py
