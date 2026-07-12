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

## Deployment

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Push this repository to GitHub.
2. Go to [Render](https://dashboard.render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Render auto-detects `render.yaml` — or configure manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn "app:create_app()"`
5. Set the environment variable `SECRET_KEY` to a secure random value.
6. Ensure `models/retina_model.keras` is present in the repository (not gitignored).

The model file (`retina_model.keras`) is tracked by Git via `.gitignore` exception. Place your trained model in `models/` before deploying.

## License

Add a license appropriate for this project.
