lint:
	black .
	isort .
	flake8 .
	mypy . --check-untyped-defs

test:
	pytest

check: lint test
