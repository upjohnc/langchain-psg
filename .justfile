default:
    just --list

install:
    poetry install --no-root

pre-commit:
    pre-commit install

run:
    PYTHONPATH=. poetry run python src/code.py

