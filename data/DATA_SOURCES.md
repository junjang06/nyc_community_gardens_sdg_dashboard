# Dashboard data files

## `nta_population.csv`

Columns:
- `nta_code`: 2020 NYC Neighborhood Tabulation Area code
- `nta_name`: 2020 NTA name
- `population`: 2020 Census population
- `source`: source note
- `source_url`: source URL

This file was compiled from the Prison Policy Initiative's NYC NTA appendix, which includes a `2020 Census population` column for NYC 2020 NTAs. The page states that the column is the population of the NTA as reported in the 2020 Census.

Source: https://www.prisonpolicy.org/origin/ny/2020/nyc_nta.html

## HVI handling

The requested NYC Open Data HVI dataset (`4mhf-duep`) is ZIP/ZCTA-based, not directly NTA-based. To avoid a misleading ZIP-to-NTA join, the patched `app.py` now attempts to load NYC DOHMH's official ArcGIS CDTA-level HVI layer at runtime, then assigns the parent CDTA score to NTAs using the first four characters of the NTA code, e.g. `BK0101 -> BK01`.

Official HVI source used by the patched app:
https://services1.arcgis.com/8cuieNI8NbqQZQVJ/ArcGIS/rest/services/HVI_by_CDTA_CRAD_2024/FeatureServer/0

This is a CDTA-to-NTA proxy. It is better than fabricated NTA HVI values, but it should still be described as an exploratory aggregation for presentation use.
