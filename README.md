# RetinaAI

An AI-based web application for diabetic retinopathy classification using the **APTOS 2019 Blindness Detection** dataset.

## Project status

Project scaffold only. No machine-learning model or training pipeline has been implemented yet.

## Dataset

Paste the APTOS 2019 dataset into `dataset/raw/`. Keep the original files intact; preprocessing outputs should be written to `dataset/processed/`.

## Directory guide

- `src/preprocessing/`: image preparation and dataset processing.
- `src/training/`: model definition and training orchestration.
- `src/prediction/`: inference workflow.
- `static/` and `templates/`: Flask web assets.
- `reports/`: generated PDF reports and figures.
- `models/`: trained model artifacts (ignored by Git).

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

## Next steps

1. Add the dataset to `dataset/raw/`.
2. Implement preprocessing and quality checks.
3. Define and train a classification model.
4. Integrate prediction into the Flask interface.

## License

Add a license appropriate for this project.
