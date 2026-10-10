# UK Road Accident Severity Prediction

Predicting whether a UK road accident is **fatal, serious or slight** from variables observable at the scene,
and testing whether neural networks significantly outperform traditional machine learning.

> 🚧 Work in progress. Live demo: _coming soon_.

## Problem
UK emergency services want to know whether accident severity can be predicted from information gathered at the
scene, to support a faster, better-matched response. Data: all reported 2019 accidents (`accident_severity` is the target).

## Approach (planned)
1. Data audit and cleaning (corrupted IDs, hidden missing labels, structured missingness)
2. EDA
3. Leakage-safe sklearn pipelines and class-imbalance strategies
4. Traditional ML: logistic regression, random forest, XGBoost, LightGBM
5. Neural network (MLP) with multi-seed evaluation
6. Statistical comparison of models (macro-F1, per-class recall, McNemar)
7. Interpretability (SHAP), calibration, error analysis, scene-only ablation
8. Streamlit app deployed free

## Results
_To be filled in._

## Limitations
_To be filled in (single year, imbalance, post-crash features, no location data)._

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place uk_road_accident_2019.csv in data/raw/
python -m src.data          # audit + clean
pytest -q                   # tests
streamlit run app/streamlit_app.py
```

## Repo structure
```
data/        raw and processed data (git-ignored)
notebooks/   analysis notebooks
src/         reusable code (config, data, features, models)
app/         Streamlit app
models/      saved pipelines
reports/     figures and write-ups
tests/       pytest tests
```
