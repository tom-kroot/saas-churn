
.PHONY: quickstart data features train evaluate app test lint

quickstart: data features train evaluate

data:
	python -m src.utils.generate_data

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

app:
	streamlit run app/Home.py

test:
	pytest -q