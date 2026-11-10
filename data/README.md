# Data

The pipeline expects one CSV file, `data/sw_nigeria_yield.csv`, with one row per state, crop and year.

| Column | Description |
|---|---|
| state | Ekiti, Lagos, Ogun, Ondo, Osun or Oyo |
| crop | e.g. Cassava, Yam, Maize, Cocoyam, Plantain |
| year | Cropping season / calendar year |
| area_harvested_ha | Area harvested (hectares) |
| yield_kg_ha | Yield (kg/ha) — the target variable |
| rainfall_mm | Growing-season rainfall total (mm) |
| tmax_c, tmin_c | Mean daily maximum / minimum temperature (°C) |
| humidity_pct | Mean relative humidity (%) |
| solar_mj_m2 | Mean daily solar radiation (MJ/m²) |

`sw_nigeria_yield_template.csv` contains the header row only.

## Sources for real data

- Yield and area: Nigeria Bureau of Statistics (https://nigerianstat.gov.ng/), State Agricultural Development Programmes, FAOSTAT (https://www.fao.org/faostat/).
- Climate: NASA POWER (https://power.larc.nasa.gov/), Nigerian Meteorological Agency (https://nimet.gov.ng/).

## Demonstration data

`python scripts/make_demo_data.py` writes a **synthetic demonstration dataset** to `data/sw_nigeria_yield.csv`
so the pipeline and web app can be run and tested. It is not real harvest data; any results produced from
it describe how the pipeline behaves, not yields in South-West Nigeria.
