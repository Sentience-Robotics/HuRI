lint:
	black .
	isort .


	mypy . --check-untyped-defs

test:
	pytest

check: lint test
