run-core:
	poetry run gradio src/stability_core.py
run-v1:
	poetry run gradio src/stability_v1.py
lint:
	poetry run flake8 src
format:
	poetry run black src
sort:
	poetry run isort src