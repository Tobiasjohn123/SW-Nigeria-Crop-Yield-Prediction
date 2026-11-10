# Machine Learning-Driven Prediction of Tropical Agricultural Produce in Nigeria
### Case study: South-West Nigeria (Ekiti, Lagos, Ogun, Ondo, Osun, Oyo)

CSC 476 semester project, Department of Computer Science, University of Ibadan.

This repository contains the software implementation for the project report. It predicts crop yield (kg/ha)
and production (tonnes) at state-crop-year level using climate and lagged-yield features, compares four
machine learning models against a persistence baseline, and serves the best model through a Streamlit web app.

## What is in the repository

| Path | Purpose |
|---|---|
| `train_models.py` | Data cleaning, feature engineering, time-ordered split, tuning, evaluation, permutation importance, model saving |
| `app.py` | Streamlit web interface that loads the saved model and returns yield and production estimates |
| `scripts/make_demo_data.py` | Writes a synthetic **demonstration** dataset so the code can be run without real data |
| `tests/test_pipeline.py` | Unit tests (lag features use only past seasons, validation never trains on the future, metrics) |
| `data/` | Expected data schema, header-only template and data sources |

## Method in brief

- Features: current-season rainfall, temperature, humidity and solar radiation; area harvested; state and crop;
  previous-season and 3-season rolling yield and rainfall.
- Models: Ridge Regression, Support Vector Regression, Random Forest, Gradient Boosting, plus a persistence baseline.
- Validation: the last three years are held out as the test set; tuning uses expanding-window cross-validation
  (train on years up to Y, validate on Y+1), so the models never train on the future.
- Metrics: RMSE, MAE, R², MAPE; permutation importance for interpretation.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Provide data: either place real records in data/sw_nigeria_yield.csv (see data/README.md),
#    or create the synthetic demonstration dataset:
python scripts/make_demo_data.py

# 2. Train, evaluate and save the best model
python train_models.py          # writes outputs/*.csv and models/best_model.joblib

# 3. Launch the web interface
streamlit run app.py

# 4. Run the tests
pytest
```

## Important note on data

No real harvest records are bundled. Results obtained from `scripts/make_demo_data.py` demonstrate that the
pipeline works; they are not findings about South-West Nigeria. Use real NBS / ADP / FAOSTAT yield records
merged with NASA POWER / NiMet climate data for real results.

## Team

Team members and their contributions are listed in the project report (front matter).

## Licence

MIT — see `LICENSE`.
