lint:
	black .
	isort .
	flake8 .
	mypy .

test:
	pytest

check: lint test