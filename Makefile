install:
	pip install uv &&\
	uv sync --no-dev

test:
	uv run --no-dev python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli

format:	
	uv run --no-dev black mylib/*.py cli/*.py api/*.py model/*.py

lint:
	uv run --no-dev pylint --disable=R,C --generated-members=cv2.* --ignore-patterns=test_.*\.py mylib/*.py cli/*.py api/*.py model/*.py

refactor: format lint

all: install refactor test