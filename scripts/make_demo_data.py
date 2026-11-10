"""
make_demo_data.py
Creates a DEMONSTRATION dataset (data/sw_nigeria_yield.csv) that follows the project schema.
It is NOT real harvest data. It exists so the pipeline and the web app can be run and tested
end-to-end. Replace data/sw_nigeria_yield.csv with real NBS / ADP / FAOSTAT yield records merged
with NASA POWER / NiMet climate data to produce real results.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
rows = []
for state in ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"]:
    for crop, base in [("Cassava", 12000), ("Yam", 10000), ("Maize", 1500)]:
        for year in range(2000, 2024):
            rain = rng.normal(1300, 200)
            rows.append(dict(
                state=state, crop=crop, year=year,
                area_harvested_ha=rng.uniform(5e3, 5e4),
                yield_kg_ha=base * (1 + 0.01 * (year - 2000)) + 0.3 * (rain - 1300) + rng.normal(0, 300),
                rainfall_mm=rain, tmax_c=rng.normal(32, 1), tmin_c=rng.normal(22, 1),
                humidity_pct=rng.normal(75, 5), solar_mj_m2=rng.normal(18, 1)))

os.makedirs("data", exist_ok=True)
pd.DataFrame(rows).to_csv("data/sw_nigeria_yield.csv", index=False)
print("Demonstration dataset written to data/sw_nigeria_yield.csv (", len(rows), "rows )")
