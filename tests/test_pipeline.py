import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_models as tm  # noqa: E402


def _toy_csv(tmp_path):
    rows = []
    for state in ["Oyo", "Ogun"]:
        for year in range(2000, 2012):
            rows.append(dict(state=state, crop="Cassava", year=year, area_harvested_ha=1000.0,
                             yield_kg_ha=10000.0 + 100 * (year - 2000), rainfall_mm=1200.0 + year,
                             tmax_c=32.0, tmin_c=22.0, humidity_pct=75.0, solar_mj_m2=18.0))
    path = tmp_path / "toy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_lag_features_use_only_previous_season(tmp_path):
    df = tm.load_and_engineer(_toy_csv(tmp_path))
    for (_, _), g in df.groupby(["state", "crop"]):
        g = g.sort_values("year")
        # yield_lag1 must equal the previous year's yield, never the current or a future one
        assert np.allclose(g["yield_lag1"].iloc[1:].to_numpy(), g[tm.TARGET].iloc[:-1].to_numpy())
        assert (g["yield_roll3"] <= g[tm.TARGET].max()).all()


def test_first_season_without_history_is_dropped(tmp_path):
    df = tm.load_and_engineer(_toy_csv(tmp_path))
    assert df["year"].min() == 2001


def test_expanding_window_never_trains_on_the_future(tmp_path):
    df = tm.load_and_engineer(_toy_csv(tmp_path))
    folds = list(tm.expanding_window_splits(df, min_train_years=4))
    assert len(folds) > 0
    for tr, va in folds:
        assert df.loc[tr, "year"].max() < df.loc[va, "year"].min()
        assert df.loc[va, "year"].nunique() == 1


def test_metrics_are_zero_for_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    m = tm.metrics(y, y)
    assert m["RMSE"] == 0 and m["MAE"] == 0 and m["R2"] == 1.0
