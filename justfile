default:
    @just --list

install:
    test -d .venv || python3 -m venv .venv
    uv sync

preprocess:
    uv run python src/ptb_xl_ecg_classification/preprocessing/create_clean_dataset.py
verify-preprocess:
    uv run python src/ptb_xl_ecg_classification/preprocessing/verify_dataset.py

train-neural-network:
    uv run python src/ptb_xl_ecg_classification/models/neural_network/train.py
evaluate-neural-network:
    uv run python src/ptb_xl_ecg_classification/models/neural_network/evaluate.py

random-forest:
    uv run python src/ptb_xl_ecg_classification/models/random_forest/train.py

run-svm: svm-feature-selection train-svm evaluate-svm
svm-feature-selection:
    uv run python -m src.ptb_xl_ecg_classification.models.support_vector_machine.feature_selection
train-svm:
    uv run python -m src.ptb_xl_ecg_classification.models.support_vector_machine.train
evaluate-svm:
    uv run python -m src.ptb_xl_ecg_classification.models.support_vector_machine.evaluate
