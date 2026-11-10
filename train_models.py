"""
train_models.py
Machine Learning-Driven Prediction of Tropical Agricultural Produce in Nigeria
Case Study: South-West, Nigeria

Input : data/sw_nigeria_yield.csv  (one row per state-crop-year)
Output: outputs/model_metrics.csv, outputs/predictions.csv,
        outputs/feature_importance.csv, models/best_model.joblib
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "data/sw_nigeria_yield.csv"
TEST_YEARS = 3            # most recent years reserved as the final test set
MIN_TRAIN_YEARS = 8       # first expanding-window fold starts with this many years

CAT_COLS = ["state", "crop"]
NUM_COLS = ["area_harvested_ha", "rainfall_mm", "tmax_c", "tmin_c",
            "humidity_pct", "solar_mj_m2", "rain_lag1", "rain_roll3",
            "yield_lag1", "yield_roll3", "year"]
TARGET = "yield_kg_ha"


# ---------------------------------------------------------------- data ---
def load_and_engineer(path=DATA_PATH):
    df = pd.read_csv(path).sort_values(["state", "crop", "year"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["state", "crop", "year"])
    df = df[df[TARGET] > 0]

    # Winsorise extreme yields within each crop (1st and 99th percentile)
    lo = df.groupby("crop")[TARGET].transform(lambda s: s.quantile(0.01))
    hi = df.groupby("crop")[TARGET].transform(lambda s: s.quantile(0.99))
    df[TARGET] = df[TARGET].clip(lo, hi)

    # Lag / rolling features are built per state-crop series (no future leakage)
    g = df.groupby(["state", "crop"])
    df["yield_lag1"] = g[TARGET].shift(1)
    df["yield_roll3"] = g[TARGET].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["rain_lag1"] = g["rainfall_mm"].shift(1)
    df["rain_roll3"] = g["rainfall_mm"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return df.dropna(subset=["yield_lag1"]).reset_index(drop=True)


def expanding_window_splits(df, min_train_years=MIN_TRAIN_YEARS, last_train_year=None):
    """Yield (train_idx, val_idx): train on all years <= Y, validate on year Y+1."""
    years = sorted(df["year"].unique())
    if last_train_year is not None:
        years = [y for y in years if y <= last_train_year]
    for i in range(min_train_years, len(years)):
        tr = df.index[df["year"] <= years[i - 1]].to_numpy()
        va = df.index[df["year"] == years[i]].to_numpy()
        yield tr, va


# -------------------------------------------------------------- models ---
def make_preprocessor(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def model_zoo():
    return {
        "Ridge Regression": (
            Pipeline([("pre", make_preprocessor(True)), ("m", Ridge())]),
            {"m__alpha": [0.1, 1, 10, 100]}),
        "Support Vector Regression": (
            Pipeline([("pre", make_preprocessor(True)),
                      ("m", TransformedTargetRegressor(regressor=SVR(kernel="rbf"),
                                                       transformer=StandardScaler()))]),
            {"m__regressor__C": [1, 10, 100], "m__regressor__gamma": ["scale", 0.01],
             "m__regressor__epsilon": [0.05, 0.1]}),
        "Random Forest": (
            Pipeline([("pre", make_preprocessor(False)),
                      ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
            {"m__n_estimators": [200, 500], "m__max_depth": [None, 8, 16],
             "m__min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (
            Pipeline([("pre", make_preprocessor(False)),
                      ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
            {"m__n_estimators": [200, 400], "m__learning_rate": [0.03, 0.1],
             "m__max_depth": [2, 3, 4]}),
    }


def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


# ---------------------------------------------------------------- main ---
def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_and_engineer()
    cutoff = sorted(df["year"].unique())[-TEST_YEARS - 1]
    train = df[df["year"] <= cutoff].reset_index(drop=True)
    test = df[df["year"] > cutoff].reset_index(drop=True)
    print(f"Train rows: {len(train)} | Test rows: {len(test)} | Test years: "
          f"{sorted(int(y) for y in test['year'].unique())}")

    X_tr, y_tr = train[CAT_COLS + NUM_COLS], train[TARGET]
    X_te, y_te = test[CAT_COLS + NUM_COLS], test[TARGET]

    rows, preds = [], test[["state", "crop", "year", TARGET]].copy()

    # Persistence baseline: this year's yield = last year's yield
    base = metrics(y_te, test["yield_lag1"])
    rows.append({"Model": "Persistence baseline", "Best params": "-", "CV_RMSE": np.nan, **base})
    preds["Persistence baseline"] = test["yield_lag1"].to_numpy()

    fitted = {}
    cv = list(expanding_window_splits(train))
    for name, (pipe, grid) in model_zoo().items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        y_hat = gs.best_estimator_.predict(X_te)
        rows.append({"Model": name, "Best params": str(gs.best_params_),
                     "CV_RMSE": -gs.best_score_, **metrics(y_te, y_hat)})
        preds[name] = y_hat
        fitted[name] = gs.best_estimator_
        print(f"{name:28s} CV-RMSE={-gs.best_score_:9.2f}  Test-RMSE={rows[-1]['RMSE']:9.2f}")

    res = pd.DataFrame(rows).sort_values("RMSE")
    res.to_csv("outputs/model_metrics.csv", index=False)
    preds.to_csv("outputs/predictions.csv", index=False)

    best_name = res[res["Model"].isin(fitted)].iloc[0]["Model"]
    best = fitted[best_name]
    print("\nBest machine-learning model:", best_name)

    # Model-agnostic interpretation on the held-out years
    pi = permutation_importance(best, X_te, y_te, n_repeats=20, random_state=RANDOM_STATE,
                                scoring="neg_root_mean_squared_error")
    imp = (pd.DataFrame({"feature": X_te.columns, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False))
    imp.to_csv("outputs/feature_importance.csv", index=False)

    joblib.dump({"model": best, "name": best_name, "features": CAT_COLS + NUM_COLS},
                "models/best_model.joblib")
    print(res[["Model", "CV_RMSE", "RMSE", "MAE", "R2", "MAPE_%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
