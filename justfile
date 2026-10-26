default:
    @just --list

install:
    test -d .venv || python3 -m venv .venv
    uv sync
