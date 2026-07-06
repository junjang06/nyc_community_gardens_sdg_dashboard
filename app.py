"""
NYC Community Gardens for Equitable Climate Resilience
Production-ready Streamlit dashboard with official NYC Open Data loading and mock-data fallbacks.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Primary data sources:
    GreenThumb Garden Info: p78i-pat6
    2020 Neighborhood Tabulation Areas: 9nt8-h7nd
    Heat Vulnerability Index Rankings: 4mhf-duep
    Optional local population file: data/nta_population.csv
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from shapely import wkt
from shapely.geometry import Point, box, shape

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NYC Community Gardens for Equitable Climate Resilience",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Constants and styling
# -----------------------------------------------------------------------------

GREENTHUMB_ID = "p78i-pat6"
NTA_ID = "9nt8-h7nd"
HVI_ID = "4mhf-duep"
POPULATION_ID = "swpk-hqdp"  # Older NYC NTA population table; local 2020 CSV is preferred.

SODA_JSON = "https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit=50000"
SODA_GEOJSON = "https://data.cityofnewyork.us/api/views/{dataset_id}/rows.geojson?accessType=DOWNLOAD"
SODA_GEOSPATIAL_EXPORT = (
    "https://data.cityofnewyork.us/api/geospatial/{dataset_id}?method=export&format=GeoJSON"
)

NYC_CENTER = {"lat": 40.7128, "lon": -74.0060}
AREA_SQFT_PER_SQMI = 27_878_400
NYC_LOCAL_CRS = "EPSG:2263"  # NAD83 / New York Long Island ftUS
WGS84 = "EPSG:4326"

PALETTE = {
    "forest": "#12372A",
    "sage": "#9CAF88",
    "cream": "#F6F2E8",
    "terracotta": "#C76F4A",
    "mint": "#DDE8D1",
    "ink": "#1E2A24",
    "muted": "#66756D",
    "white": "#FFFFFF",
}

BOROUGH_ORDER = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {PALETTE['cream']};
            color: {PALETTE['ink']};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #12372A 0%, #1E4A39 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #F6F2E8 !important;
        }}
        .main-title {{
            font-size: 2.55rem;
            line-height: 1.05;
            font-weight: 850;
            color: {PALETTE['forest']};
            margin-bottom: 0.15rem;
        }}
        .guiding-question {{
            font-size: 1.28rem;
            font-weight: 650;
            color: {PALETTE['terracotta']};
            background: #FFF8EF;
            border-left: 6px solid {PALETTE['terracotta']};
            padding: 1rem 1.1rem;
            border-radius: 0.65rem;
            margin: 0.6rem 0 1.0rem 0;
        }}
        .subtle {{
            color: {PALETTE['muted']};
            font-size: 0.96rem;
        }}
        .kpi-card {{
            background: {PALETTE['white']};
            border: 1px solid #E6E1D5;
            border-radius: 1rem;
            padding: 1rem 1.05rem;
            box-shadow: 0 4px 14px rgba(18, 55, 42, 0.08);
            min-height: 118px;
        }}
        .kpi-label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: {PALETTE['muted']};
            font-weight: 750;
        }}
        .kpi-value {{
            font-size: 2rem;
            color: {PALETTE['forest']};
            font-weight: 850;
            margin-top: 0.25rem;
        }}
        .callout {{
            background: {PALETTE['mint']};
            border-left: 6px solid {PALETTE['forest']};
            border-radius: 0.75rem;
            padding: 0.95rem 1.05rem;
            margin: 0.75rem 0;
            color: {PALETTE['ink']};
        }}
        .warning-callout {{
            background: #FFF1E8;
            border-left: 6px solid {PALETTE['terracotta']};
            border-radius: 0.75rem;
            padding: 0.95rem 1.05rem;
            margin: 0.75rem 0;
            color: {PALETTE['ink']};
        }}
        .action-card {{
            background: #FFFFFF;
            border: 1px solid #E6E1D5;
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 12px rgba(18, 55, 42, 0.07);
            min-height: 178px;
        }}
        .badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.32rem 0.68rem;
            margin: 0.12rem 0.2rem 0.12rem 0;
            background: {PALETTE['forest']};
            color: #FFFFFF;
            font-weight: 750;
            font-size: 0.82rem;
        }}
        .sdg-card {{
            background: #FFFFFF;
            border-radius: 1rem;
            border: 1px solid #E6E1D5;
            padding: 1rem;
            min-height: 172px;
            box-shadow: 0 4px 12px rgba(18, 55, 42, 0.06);
        }}
        div[data-testid="stMetric"] {{
            background: white;
            padding: 0.85rem;
            border-radius: 0.9rem;
            border: 1px solid #E6E1D5;
        }}
        .block-container {{padding-top: 2rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def normalize_col_name(col: str) -> str:
    """Normalize raw Socrata/CSV column names for safer matching."""
    return re.sub(r"[^a-z0-9_]+", "_", str(col).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        c_norm = normalize_col_name(c)
        if c_norm in cols:
            return c_norm
    return None


def normalize_text_key(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def request_json(url: str, timeout: int = 25) -> List[Dict]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "error" in data:
        raise ValueError(data.get("message", data["error"]))
    if not isinstance(data, list):
        raise ValueError("Expected a list of JSON records from Socrata API.")
    return data


def get_borough_options(gdf: gpd.GeoDataFrame) -> List[str]:
    if "borough" not in gdf.columns:
        return ["All boroughs"]
    found = [b for b in BOROUGH_ORDER if b in set(gdf["borough"].dropna().astype(str))]
    extra = sorted(set(gdf["borough"].dropna().astype(str)) - set(found))
    return ["All boroughs"] + found + extra


def filtered_by_borough(gdf: gpd.GeoDataFrame, borough: str) -> gpd.GeoDataFrame:
    if borough == "All boroughs" or "borough" not in gdf.columns:
        return gdf.copy()
    return gdf[gdf["borough"].astype(str) == borough].copy()


def friendly_metric_name(metric: str) -> str:
    names = {
        "garden_count": "Garden Count",
        "garden_access_score": "Gardens per 10,000 Residents",
        "garden_density_per_sq_mile": "Gardens per Square Mile",
        "hvi_rank": "Heat Vulnerability Index",
    }
    return names.get(metric, metric.replace("_", " ").title())


def city_median(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if pd.isna(value):
        return 0.0
    return float(value)


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> Dict:
    """Convert a GeoDataFrame to a JSON-serializable GeoJSON dict for Plotly."""
    return json.loads(gdf.to_json())


def standardize_borough(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    mapping = {
        "1": "Manhattan",
        "2": "Bronx",
        "3": "Brooklyn",
        "4": "Queens",
        "5": "Staten Island",
        "mn": "Manhattan",
        "manhattan": "Manhattan",
        "new york": "Manhattan",
        "bx": "Bronx",
        "bronx": "Bronx",
        "bk": "Brooklyn",
        "kings": "Brooklyn",
        "brooklyn": "Brooklyn",
        "qn": "Queens",
        "queens": "Queens",
        "si": "Staten Island",
        "staten island": "Staten Island",
        "richmond": "Staten Island",
    }
    return mapping.get(v, str(value).strip())


# -----------------------------------------------------------------------------
# Mock data fallbacks
# -----------------------------------------------------------------------------


def create_mock_nta_data() -> gpd.GeoDataFrame:
    """Create deterministic simplified NTA-like polygons for offline demos."""
    borough_specs = {
        "Bronx": (-73.91, 40.83, 0.045, 0.035),
        "Brooklyn": (-73.96, 40.64, 0.055, 0.04),
        "Manhattan": (-73.99, 40.76, 0.035, 0.035),
        "Queens": (-73.82, 40.71, 0.06, 0.04),
        "Staten Island": (-74.15, 40.58, 0.06, 0.045),
    }
    records = []
    idx = 1
    for borough, (lon0, lat0, dx, dy) in borough_specs.items():
        for row in range(2):
            for col in range(2):
                lon = lon0 + col * dx
                lat = lat0 + row * dy
                records.append(
                    {
                        "nta_code": f"MOCK{idx:03d}",
                        "nta_name": f"Mock {borough} Resilience Area {row * 2 + col + 1}",
                        "borough": borough,
                        "geometry": box(lon, lat, lon + dx * 0.86, lat + dy * 0.86),
                        "mock_nta": True,
                    }
                )
                idx += 1
    return gpd.GeoDataFrame(records, geometry="geometry", crs=WGS84)


def create_mock_garden_data() -> gpd.GeoDataFrame:
    """Create deterministic garden points clustered by borough for offline demos."""
    random.seed(42)
    centers = {
        "Bronx": (-73.90, 40.84, 24),
        "Brooklyn": (-73.94, 40.67, 42),
        "Manhattan": (-73.97, 40.78, 28),
        "Queens": (-73.83, 40.72, 26),
        "Staten Island": (-74.15, 40.59, 10),
    }
    rows = []
    garden_id = 1
    for borough, (lon, lat, n) in centers.items():
        for i in range(n):
            rows.append(
                {
                    "garden_id": f"MOCK-G{garden_id:04d}",
                    "garden_name": f"Mock {borough} Community Garden {i + 1}",
                    "borough": borough,
                    "status": "Active" if i % 7 != 0 else "Other / Unknown",
                    "address": "Mock address for offline fallback",
                    "mock_garden": True,
                    "geometry": Point(
                        lon + random.uniform(-0.045, 0.045),
                        lat + random.uniform(-0.035, 0.035),
                    ),
                }
            )
            garden_id += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)


def create_mock_population_data(nta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create deterministic population placeholders keyed to NTA code."""
    rng = np.random.default_rng(7)
    df = nta_gdf[["nta_code", "nta_name", "borough"]].copy()
    borough_base = {
        "Bronx": 41000,
        "Brooklyn": 52000,
        "Manhattan": 61000,
        "Queens": 48000,
        "Staten Island": 33000,
    }
    df["population"] = [
        int(borough_base.get(b, 45000) + rng.integers(-12000, 15000)) for b in df["borough"]
    ]
    df["population_source"] = "mock_placeholder"
    return df


def create_mock_hvi_data(nta_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create deterministic HVI placeholders keyed to NTA code."""
    rng = np.random.default_rng(11)
    base = {"Bronx": 4, "Brooklyn": 3, "Manhattan": 2, "Queens": 3, "Staten Island": 2}
    hvi = nta_gdf[["nta_code", "nta_name", "borough"]].copy()
    hvi["hvi_rank"] = [
        int(np.clip(base.get(b, 3) + rng.choice([-1, 0, 0, 1]), 1, 5)) for b in hvi["borough"]
    ]
    hvi["hvi_source"] = "mock_placeholder"
    return hvi


# -----------------------------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_greenthumb_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load official NYC GreenThumb Garden Info data.

    The official dataset may include polygon/multipolygon geometry instead of lat/lon.
    clean_garden_points() converts whatever geometry exists into point locations.
    """
    status = {
        "dataset": "GreenThumb Garden Info",
        "dataset_id": GREENTHUMB_ID,
        "source": "NYC Open Data",
        "mock": "False",
        "message": "Loaded from NYC Open Data GeoJSON/API.",
    }

    geojson_url = SODA_GEOJSON.format(dataset_id=GREENTHUMB_ID)
    json_url = SODA_JSON.format(dataset_id=GREENTHUMB_ID)

    try:
        gdf = gpd.read_file(geojson_url)
        if len(gdf) == 0:
            raise ValueError("GeoJSON returned no rows.")
        gdf = normalize_columns(gdf)
        return gdf, status
    except Exception as geo_error:
        try:
            records = request_json(json_url)
            df = normalize_columns(pd.DataFrame(records))
            status["message"] = f"Loaded from NYC Open Data JSON after GeoJSON fallback: {geo_error}"
            return df, status
        except Exception as json_error:
            mock = create_mock_garden_data()
            status["mock"] = "True"
            status["source"] = "Mock fallback"
            status[
                "message"
            ] = f"Live GreenThumb data unavailable. Using mock fallback. Error: {json_error}"
            return mock, status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_nta_boundaries() -> Tuple[gpd.GeoDataFrame, Dict[str, str]]:
    """Load official 2020 NTA boundaries from NYC Open Data."""
    status = {
        "dataset": "2020 Neighborhood Tabulation Areas",
        "dataset_id": NTA_ID,
        "source": "NYC Open Data",
        "mock": "False",
        "message": "Loaded from NYC Open Data geospatial export.",
    }

    urls = [
        SODA_GEOSPATIAL_EXPORT.format(dataset_id=NTA_ID),
        SODA_GEOJSON.format(dataset_id=NTA_ID),
    ]
    last_error = None
    for url in urls:
        try:
            gdf = gpd.read_file(url)
            if len(gdf) == 0:
                raise ValueError("NTA endpoint returned no rows.")
            gdf = normalize_columns(gdf)

            code_col = first_existing(
                gdf,
                [
                    "nta2020",
                    "nta2020_code",
                    "ntacode",
                    "nta_code",
                    "geoid",
                    "nta",
                    "ntacode2020",
                ],
            )
            name_col = first_existing(
                gdf,
                [
                    "ntaname",
                    "nta_name",
                    "nta2020_name",
                    "name",
                    "neighborhood",
                    "ntaname2020",
                ],
            )
            boro_col = first_existing(
                gdf,
                [
                    "boroname",
                    "boro_name",
                    "borough",
                    "boro",
                    "borocode",
                    "boro_code",
                ],
            )

            if code_col is None:
                gdf["nta_code"] = [f"NTA_{i:04d}" for i in range(len(gdf))]
            else:
                gdf["nta_code"] = gdf[code_col].astype(str)

            if name_col is None:
                gdf["nta_name"] = gdf["nta_code"]
            else:
                gdf["nta_name"] = gdf[name_col].astype(str)

            if boro_col is None:
                gdf["borough"] = "Unknown"
            else:
                gdf["borough"] = gdf[boro_col].apply(standardize_borough)

            if gdf.crs is None:
                gdf = gdf.set_crs(WGS84)
            else:
                gdf = gdf.to_crs(WGS84)

            keep = ["nta_code", "nta_name", "borough", "geometry"]
            gdf = gdf[keep].copy()
            gdf["mock_nta"] = False
            return gdf, status
        except Exception as exc:
            last_error = exc

    mock = create_mock_nta_data()
    status["mock"] = "True"
    status["source"] = "Mock fallback"
    status["message"] = f"Live NTA boundaries unavailable. Using mock fallback. Error: {last_error}"
    return mock, status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_hvi_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load official Heat Vulnerability Index Rankings from NYC Open Data."""
    status = {
        "dataset": "Heat Vulnerability Index Rankings",
        "dataset_id": HVI_ID,
        "source": "NYC Open Data",
        "mock": "False",
        "message": "Loaded from NYC Open Data JSON.",
    }
    try:
        records = request_json(SODA_JSON.format(dataset_id=HVI_ID))
        df = normalize_columns(pd.DataFrame(records))
        if len(df) == 0:
            raise ValueError("HVI endpoint returned no rows.")
        return df, status
    except Exception as exc:
        status["mock"] = "True"
        status["source"] = "Mock fallback pending NTA join"
        status["message"] = f"Live HVI data unavailable. Will use mock HVI fallback. Error: {exc}"
        return pd.DataFrame(), status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_population_data(nta_reference: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load population denominators.

    Preferred production path:
        Place a clean CSV at data/nta_population.csv with columns:
            nta_code, nta_name, population

    The older NYC Open Data population table is attempted as a secondary source, but it may
    use older NTA definitions. If neither source works, mock placeholders are generated.
    """
    status = {
        "dataset": "NTA population",
        "dataset_id": "local CSV preferred / swpk-hqdp fallback",
        "source": "Local CSV or NYC Open Data fallback",
        "mock": "False",
        "message": "Loaded population data.",
    }

    local_paths = ["data/nta_population.csv", "nta_population.csv"]
    for path in local_paths:
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            pop_col = first_existing(df, ["population", "pop", "total_population", "pop_2020"])
            code_col = first_existing(df, ["nta_code", "ntacode", "nta2020", "geoid"])
            name_col = first_existing(df, ["nta_name", "ntaname", "nta2020_name", "name"])
            if pop_col is None or (code_col is None and name_col is None):
                raise ValueError("Population CSV needs population plus nta_code or nta_name.")
            out = pd.DataFrame()
            if code_col:
                out["nta_code"] = df[code_col].astype(str)
            if name_col:
                out["nta_name"] = df[name_col].astype(str)
            out["population"] = pd.to_numeric(df[pop_col], errors="coerce")
            out = out.dropna(subset=["population"])
            out["population_source"] = path
            status["source"] = path
            status["message"] = f"Loaded local population file: {path}"
            return out, status
        except Exception:
            continue

    # Optional API fallback. This is not ideal for 2020 NTA boundaries, but useful for prototyping.
    try:
        records = request_json(SODA_JSON.format(dataset_id=POPULATION_ID))
        df = normalize_columns(pd.DataFrame(records))
        pop_col = first_existing(
            df,
            [
                "population",
                "population_2010",
                "pop_2010",
                "total_population",
                "p0010001",
                "count",
            ],
        )
        code_col = first_existing(df, ["nta_code", "ntacode", "nta", "geoid"])
        name_col = first_existing(df, ["nta_name", "ntaname", "name"])
        if pop_col is None or (code_col is None and name_col is None):
            raise ValueError("Could not identify population columns in API fallback.")
        out = pd.DataFrame()
        if code_col:
            out["nta_code"] = df[code_col].astype(str)
        if name_col:
            out["nta_name"] = df[name_col].astype(str)
        out["population"] = pd.to_numeric(df[pop_col], errors="coerce")
        out = out.dropna(subset=["population"])
        out["population_source"] = f"NYC Open Data {POPULATION_ID}; verify boundary year"
        status["source"] = f"NYC Open Data {POPULATION_ID}; verify boundary year"
        status[
            "message"
        ] = "Loaded API fallback population table. Verify NTA year compatibility before formal use."
        return out, status
    except Exception as exc:
        if nta_reference is not None and len(nta_reference) > 0:
            mock = create_mock_population_data(nta_reference)
        else:
            mock = pd.DataFrame(
                {
                    "nta_code": [],
                    "nta_name": [],
                    "population": [],
                    "population_source": [],
                }
            )
        status["mock"] = "True"
        status["source"] = "Mock fallback"
        status["message"] = f"Population data unavailable. Using mock placeholders. Error: {exc}"
        return mock, status


# -----------------------------------------------------------------------------
# Cleaning and metric engineering functions
# -----------------------------------------------------------------------------


def clean_garden_points(raw_data: pd.DataFrame | gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict[str, str]]:
    """
    Convert raw garden records into a point GeoDataFrame.

    Logic:
    1. Use existing GeoDataFrame geometry when available.
    2. If geometry is polygon/multipolygon, use representative points for mapping and joins.
    3. If latitude/longitude columns exist, build points.
    4. If WKT geometry columns exist, parse them.
    5. If none exist, return mock sample and clearly mark it as mock.
    """
    status = {
        "mock": "False",
        "message": "Garden locations cleaned from available geometry/coordinates.",
    }

    if raw_data is None or len(raw_data) == 0:
        status["mock"] = "True"
        status["message"] = "No garden data supplied. Using mock garden points."
        return create_mock_garden_data(), status

    df = normalize_columns(raw_data)

    # 1. Existing GeoDataFrame geometry
    if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns and df.geometry.notna().any():
        gdf = df.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)
        # Convert any geometry type to a point representation.
        try:
            metric = gdf.to_crs(NYC_LOCAL_CRS)
            point_geom = metric.geometry.representative_point()
            gdf = gpd.GeoDataFrame(
                gdf.drop(columns=["geometry"]), geometry=point_geom, crs=NYC_LOCAL_CRS
            ).to_crs(WGS84)
        except Exception:
            gdf = gdf.to_crs(WGS84)
            gdf["geometry"] = gdf.geometry.centroid
        cleaned = standardize_garden_columns(gdf)
        return cleaned, status

    # 2. Latitude/longitude columns
    lat_col = first_existing(df, ["latitude", "lat", "y", "garden_latitude"])
    lon_col = first_existing(df, ["longitude", "lon", "lng", "long", "x", "garden_longitude"])
    if lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        valid = lat.between(40.45, 40.95) & lon.between(-74.35, -73.65)
        if valid.any():
            gdf = gpd.GeoDataFrame(
                df.loc[valid].copy(), geometry=gpd.points_from_xy(lon[valid], lat[valid]), crs=WGS84
            )
            cleaned = standardize_garden_columns(gdf)
            return cleaned, status

    # 3. WKT geometry columns
    geom_col = first_existing(df, ["multipolygon", "polygon", "the_geom", "geom", "geometry"])
    if geom_col:
        parsed = []
        for value in df[geom_col]:
            geom = None
            try:
                if isinstance(value, str) and any(
                    token in value.upper() for token in ["POINT", "POLYGON", "MULTIPOLYGON"]
                ):
                    geom = wkt.loads(value)
                elif isinstance(value, dict):
                    geom = shape(value)
            except Exception:
                geom = None
            parsed.append(geom)
        if any(g is not None for g in parsed):
            gdf = gpd.GeoDataFrame(df.copy(), geometry=parsed, crs=WGS84).dropna(subset=["geometry"])
            metric = gdf.to_crs(NYC_LOCAL_CRS)
            point_geom = metric.geometry.representative_point()
            gdf = gpd.GeoDataFrame(
                gdf.drop(columns=["geometry"]), geometry=point_geom, crs=NYC_LOCAL_CRS
            ).to_crs(WGS84)
            cleaned = standardize_garden_columns(gdf)
            return cleaned, status

    status["mock"] = "True"
    status[
        "message"
    ] = "Latitude/longitude and geometry were missing or unreadable. Using mock garden points."
    return create_mock_garden_data(), status


def standardize_garden_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create consistent garden_name, borough, status, address fields."""
    gdf = normalize_columns(gdf)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    else:
        gdf = gdf.to_crs(WGS84)

    name_col = first_existing(
        gdf,
        [
            "gardenname",
            "garden_name",
            "name",
            "site_name",
            "garden",
            "park_name",
            "propertyname",
        ],
    )
    boro_col = first_existing(gdf, ["borough", "boro", "boroname", "boro_name", "county"])
    status_col = first_existing(gdf, ["status", "garden_status", "active", "open_to_public"])
    address_col = first_existing(gdf, ["address", "location", "street_address", "cross_streets"])
    id_col = first_existing(gdf, ["garden_id", "gispropnum", "propid", "objectid", "id"])

    gdf["garden_id"] = gdf[id_col].astype(str) if id_col else [f"G{i:04d}" for i in range(len(gdf))]
    gdf["garden_name"] = gdf[name_col].astype(str) if name_col else gdf["garden_id"]
    gdf["borough"] = gdf[boro_col].apply(standardize_borough) if boro_col else None
    gdf["status"] = gdf[status_col].astype(str).replace({"nan": "Unknown"}) if status_col else "Unknown"
    gdf["address"] = gdf[address_col].astype(str).replace({"nan": ""}) if address_col else ""
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    if "mock_garden" not in gdf.columns:
        gdf["mock_garden"] = False
    return gdf


def compute_nta_metrics(
    garden_points: gpd.GeoDataFrame,
    nta_boundaries: gpd.GeoDataFrame,
    population_df: pd.DataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, str]]:
    """
    Spatially join garden points to NTAs and calculate access metrics.

    Metrics:
        Garden Count by NTA
        Garden Access Score = gardens per 10,000 residents
        Garden Density = gardens per square mile
    """
    status = {"message": "Computed NTA access metrics.", "population_mock": "False"}

    nta = nta_boundaries.copy()
    gardens = garden_points.copy()
    if nta.crs is None:
        nta = nta.set_crs(WGS84)
    if gardens.crs is None:
        gardens = gardens.set_crs(WGS84)
    nta = nta.to_crs(WGS84)
    gardens = gardens.to_crs(WGS84)

    # Area calculation in a local projected CRS.
    try:
        nta["area_sq_mi"] = nta.to_crs(NYC_LOCAL_CRS).geometry.area / AREA_SQFT_PER_SQMI
    except Exception:
        nta["area_sq_mi"] = np.nan

    # Spatial join. Fallback to manual point-in-polygon if spatial index fails.
    try:
        joined = gpd.sjoin(
            gardens[["garden_id", "garden_name", "borough", "status", "geometry"]],
            nta[["nta_code", "nta_name", "borough", "geometry"]].rename(
                columns={"borough": "nta_borough"}
            ),
            how="left",
            predicate="within",
        )
        counts = joined.groupby("nta_code", dropna=False).size().rename("garden_count")
    except Exception:
        nta_lookup = []
        for _, garden in gardens.iterrows():
            code = np.nan
            name = np.nan
            boro = np.nan
            for _, area in nta.iterrows():
                if area.geometry.contains(garden.geometry):
                    code = area["nta_code"]
                    name = area["nta_name"]
                    boro = area["borough"]
                    break
            nta_lookup.append({"nta_code": code, "nta_name": name, "nta_borough": boro})
        joined = pd.concat([gardens.reset_index(drop=True), pd.DataFrame(nta_lookup)], axis=1)
        counts = joined.groupby("nta_code", dropna=False).size().rename("garden_count")

    nta = nta.merge(counts, how="left", left_on="nta_code", right_index=True)
    nta["garden_count"] = nta["garden_count"].fillna(0).astype(int)

    # If garden records lacked borough, assign borough from NTA join when available.
    if "nta_borough" in joined.columns:
        joined["borough"] = joined["borough"].combine_first(joined["nta_borough"])

    pop = normalize_columns(population_df.copy()) if population_df is not None else pd.DataFrame()
    if len(pop) == 0:
        pop = create_mock_population_data(nta)
        status["population_mock"] = "True"
    if "population_source" in pop.columns and pop["population_source"].astype(str).str.contains("mock", case=False).any():
        status["population_mock"] = "True"

    pop_code = first_existing(pop, ["nta_code", "ntacode", "nta2020", "geoid"])
    pop_name = first_existing(pop, ["nta_name", "ntaname", "nta2020_name", "name"])
    pop_col = first_existing(pop, ["population", "pop", "total_population", "pop_2020"])

    if pop_col is None:
        pop = create_mock_population_data(nta)
        pop_code = "nta_code"
        pop_name = "nta_name"
        pop_col = "population"
        status["population_mock"] = "True"

    if pop_code:
        pop_merge = pop[[pop_code, pop_col]].rename(columns={pop_code: "nta_code", pop_col: "population"})
        nta = nta.merge(pop_merge, how="left", on="nta_code")
    elif pop_name:
        pop_merge = pop[[pop_name, pop_col]].rename(columns={pop_name: "nta_name", pop_col: "population"})
        pop_merge["nta_name_key"] = pop_merge["nta_name"].map(normalize_text_key)
        nta["nta_name_key"] = nta["nta_name"].map(normalize_text_key)
        nta = nta.merge(pop_merge[["nta_name_key", "population"]], how="left", on="nta_name_key")
    else:
        nta["population"] = np.nan

    if nta["population"].isna().mean() > 0.25:
        mock_pop = create_mock_population_data(nta)
        nta = nta.drop(columns=["population"], errors="ignore").merge(
            mock_pop[["nta_code", "population"]], on="nta_code", how="left"
        )
        status["population_mock"] = "True"
        status[
            "message"
        ] = "Population join was incomplete. Mock placeholders were used for population denominators."

    nta["population"] = pd.to_numeric(nta["population"], errors="coerce").fillna(0)
    nta["garden_access_score"] = np.where(
        nta["population"] > 0, nta["garden_count"] / nta["population"] * 10_000, 0
    )
    nta["garden_density_per_sq_mile"] = np.where(
        nta["area_sq_mi"] > 0, nta["garden_count"] / nta["area_sq_mi"], 0
    )
    nta["access_below_city_median"] = nta["garden_access_score"] < city_median(
        nta["garden_access_score"]
    )

    return nta, gpd.GeoDataFrame(joined, geometry="geometry", crs=WGS84), status


def prepare_hvi_for_join(hvi_df: pd.DataFrame) -> pd.DataFrame:
    """Create standardized hvi_rank and possible join keys from raw HVI table."""
    if hvi_df is None or len(hvi_df) == 0:
        return pd.DataFrame()
    hvi = normalize_columns(hvi_df.copy())
    rank_col = first_existing(
        hvi,
        [
            "hvi_rank",
            "hvi",
            "heat_vulnerability_index",
            "heat_vulnerability",
            "heat_vulnerability_ranking",
            "rank",
            "ranking",
            "score",
        ],
    )
    if rank_col is None:
        # Try any column containing hvi or vulnerability that is numeric-like.
        for c in hvi.columns:
            if "hvi" in c or "vulner" in c or "rank" in c:
                converted = pd.to_numeric(hvi[c], errors="coerce")
                if converted.notna().sum() > 0:
                    rank_col = c
                    break
    if rank_col is None:
        return pd.DataFrame()

    hvi["hvi_rank"] = pd.to_numeric(hvi[rank_col], errors="coerce")
    hvi = hvi.dropna(subset=["hvi_rank"]).copy()
    hvi["hvi_rank"] = hvi["hvi_rank"].clip(1, 5)

    code_col = first_existing(hvi, ["nta_code", "ntacode", "nta2020", "nta", "geoid"])
    name_col = first_existing(
        hvi,
        [
            "nta_name",
            "ntaname",
            "neighborhood",
            "area_name",
            "name",
            "neighborhood_name",
            "geo_entity_name",
        ],
    )
    boro_col = first_existing(hvi, ["borough", "boro", "boroname", "boro_name"])

    out = hvi.copy()
    if code_col:
        out["nta_code"] = out[code_col].astype(str)
    if name_col:
        out["nta_name"] = out[name_col].astype(str)
        out["nta_name_key"] = out["nta_name"].map(normalize_text_key)
    if boro_col:
        out["borough"] = out[boro_col].apply(standardize_borough)
    out["hvi_source"] = f"NYC Open Data {HVI_ID}"
    return out


def classify_priority_areas(
    nta_metrics: gpd.GeoDataFrame,
    hvi_df: pd.DataFrame,
    hvi_threshold: int = 4,
) -> Tuple[gpd.GeoDataFrame, Dict[str, str]]:
    """
    Join HVI rankings to NTA metrics and create priority classification.

    Default matrix:
        High priority   = HVI >= 4 and access below city median
        Medium priority = HVI >= 3 and access below city median
        Lower priority  = all other areas
    """
    status = {
        "hvi_mock": "False",
        "message": "Joined HVI data to NTA metrics.",
        "join_method": "none",
    }
    gdf = nta_metrics.copy()
    hvi = prepare_hvi_for_join(hvi_df)

    if len(hvi) > 0:
        joined = False
        if "nta_code" in hvi.columns:
            hvi_code = hvi[["nta_code", "hvi_rank", "hvi_source"]].drop_duplicates("nta_code")
            gdf = gdf.merge(hvi_code, on="nta_code", how="left")
            joined = gdf["hvi_rank"].notna().mean() >= 0.25
            status["join_method"] = "nta_code"
        if (not joined) and "nta_name_key" in hvi.columns:
            gdf = gdf.drop(columns=["hvi_rank", "hvi_source"], errors="ignore")
            gdf["nta_name_key"] = gdf["nta_name"].map(normalize_text_key)
            hvi_name = hvi[["nta_name_key", "hvi_rank", "hvi_source"]].drop_duplicates("nta_name_key")
            gdf = gdf.merge(hvi_name, on="nta_name_key", how="left")
            joined = gdf["hvi_rank"].notna().mean() >= 0.25
            status["join_method"] = "nta_name"

        if not joined:
            gdf = gdf.drop(columns=["hvi_rank", "hvi_source"], errors="ignore")
            mock_hvi = create_mock_hvi_data(gdf)
            gdf = gdf.merge(mock_hvi[["nta_code", "hvi_rank", "hvi_source"]], on="nta_code", how="left")
            status["hvi_mock"] = "True"
            status["join_method"] = "mock_by_nta"
            status[
                "message"
            ] = "HVI data loaded, but join keys did not align sufficiently with NTA boundaries. Mock HVI placeholders were used."
    else:
        mock_hvi = create_mock_hvi_data(gdf)
        gdf = gdf.merge(mock_hvi[["nta_code", "hvi_rank", "hvi_source"]], on="nta_code", how="left")
        status["hvi_mock"] = "True"
        status["join_method"] = "mock_by_nta"
        status["message"] = "HVI data unavailable or unreadable. Mock HVI placeholders were used."

    median_access = city_median(gdf["garden_access_score"])
    gdf["access_below_city_median"] = gdf["garden_access_score"] < median_access

    high_cut = int(hvi_threshold)
    medium_cut = max(3, high_cut - 1)

    conditions = [
        (gdf["hvi_rank"] >= high_cut) & (gdf["access_below_city_median"]),
        (gdf["hvi_rank"] >= medium_cut) & (gdf["access_below_city_median"]),
    ]
    choices = ["High priority", "Medium priority"]
    gdf["priority_class"] = np.select(conditions, choices, default="Lower priority")
    gdf["priority_sort"] = gdf["priority_class"].map(
        {"High priority": 1, "Medium priority": 2, "Lower priority": 3}
    )
    return gdf, status


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def render_kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="subtle">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(text: str, warning: bool = False) -> None:
    klass = "warning-callout" if warning else "callout"
    st.markdown(f"<div class='{klass}'>{text}</div>", unsafe_allow_html=True)


def make_garden_pydeck(gardens: gpd.GeoDataFrame, buffer_miles: float = 0.25) -> pdk.Deck:
    df = gardens.copy()
    if len(df) == 0:
        df = create_mock_garden_data()
    df["lat"] = df.geometry.y
    df["lon"] = df.geometry.x
    radius_m = max(50, float(buffer_miles) * 1609.34)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df[["garden_name", "borough", "status", "lat", "lon"]],
        get_position="[lon, lat]",
        get_radius=radius_m,
        radius_min_pixels=2,
        radius_max_pixels=35,
        get_fill_color=[18, 55, 42, 90],
        get_line_color=[199, 111, 74, 180],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=NYC_CENTER["lat"], longitude=NYC_CENTER["lon"], zoom=9.4)
    tooltip = {
        "html": "<b>{garden_name}</b><br/>Borough: {borough}<br/>Status: {status}",
        "style": {"backgroundColor": "#12372A", "color": "white"},
    }
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light")


def make_choropleth(
    gdf: gpd.GeoDataFrame,
    metric: str,
    title: str,
    color_scale: str = "YlGn",
    overlay_points: Optional[gpd.GeoDataFrame] = None,
) -> go.Figure:
    plot_gdf = gdf.copy().reset_index(drop=True)
    if len(plot_gdf) == 0 or metric not in plot_gdf.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} — no records under current filter",
            height=420,
            paper_bgcolor=PALETTE["cream"],
            plot_bgcolor=PALETTE["cream"],
            margin={"r": 0, "t": 45, "l": 0, "b": 0},
        )
        return fig

    plot_gdf["map_id"] = plot_gdf.index.astype(str)
    geojson = gdf_to_geojson(plot_gdf)

    hover_data = {
        "borough": True,
        "garden_count": True,
        "garden_access_score": ":.2f",
        "garden_density_per_sq_mile": ":.2f",
        "map_id": False,
    }
    if "hvi_rank" in plot_gdf.columns:
        hover_data["hvi_rank"] = True
    if "priority_class" in plot_gdf.columns:
        hover_data["priority_class"] = True

    fig = px.choropleth_mapbox(
        plot_gdf,
        geojson=geojson,
        locations="map_id",
        featureidkey="properties.map_id",
        color=metric,
        hover_name="nta_name",
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        mapbox_style="carto-positron",
        center=NYC_CENTER,
        zoom=9.15,
        opacity=0.68,
        title=title,
    )

    if overlay_points is not None and len(overlay_points) > 0:
        points = overlay_points.to_crs(WGS84).copy()
        points["lat"] = points.geometry.y
        points["lon"] = points.geometry.x
        fig.add_trace(
            go.Scattermapbox(
                lat=points["lat"],
                lon=points["lon"],
                mode="markers",
                marker={"size": 6, "color": PALETTE["forest"], "opacity": 0.72},
                text=points.get("garden_name", "Community garden"),
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Community gardens",
            )
        )

    fig.update_layout(
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        height=610,
        paper_bgcolor=PALETTE["cream"],
        plot_bgcolor=PALETTE["cream"],
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.01),
    )
    return fig


def priority_download_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [
        "priority_class",
        "borough",
        "nta_code",
        "nta_name",
        "hvi_rank",
        "garden_count",
        "population",
        "garden_access_score",
        "garden_density_per_sq_mile",
    ]
    available = [c for c in cols if c in gdf.columns]
    return gdf[available].sort_values(["priority_class", "garden_access_score", "hvi_rank"]).copy()


# -----------------------------------------------------------------------------
# App data pipeline
# -----------------------------------------------------------------------------


with st.spinner("Loading NYC Open Data and building resilience metrics..."):
    raw_gardens, greenthumb_status = load_greenthumb_data()
    nta_boundaries, nta_status = load_nta_boundaries()
    hvi_raw, hvi_load_status = load_hvi_data()
    population_raw, population_status = load_population_data(nta_boundaries)
    gardens, garden_clean_status = clean_garden_points(raw_gardens)
    nta_metrics, garden_joined, metrics_status = compute_nta_metrics(
        gardens, nta_boundaries, population_raw
    )
    priority_gdf, hvi_join_status = classify_priority_areas(nta_metrics, hvi_raw, hvi_threshold=4)

# Use spatially derived borough for gardens when raw garden borough is missing.
if "nta_borough" in garden_joined.columns:
    gardens = garden_joined.copy()
    if "borough" not in gardens.columns:
        gardens["borough"] = gardens["nta_borough"]
    else:
        gardens["borough"] = gardens["borough"].combine_first(gardens["nta_borough"])
    gardens["lat"] = gardens.geometry.y
    gardens["lon"] = gardens.geometry.x

# -----------------------------------------------------------------------------
# Header and sidebar
# -----------------------------------------------------------------------------

st.markdown(
    "<div class='main-title'>NYC Community Gardens for Equitable Climate Resilience</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='guiding-question'>How can NYC community gardens support sustainable, equitable, and climate-resilient cities?</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtle'>A youth-friendly civic-tech dashboard connecting community gardens to SDG 11, SDG 13, SDG 15, and supporting SDG 2 and SDG 3.</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Global filters")
borough_choice = st.sidebar.selectbox("Borough", get_borough_options(priority_gdf), index=0)
metric_view = st.sidebar.selectbox(
    "Metric view",
    ["garden_access_score", "garden_density_per_sq_mile", "garden_count"],
    format_func=friendly_metric_name,
    index=0,
)
buffer_distance = st.sidebar.slider(
    "Garden influence buffer / visual radius", min_value=0.10, max_value=1.00, value=0.25, step=0.05
)
hvi_threshold = st.sidebar.slider("High HVI threshold", min_value=1, max_value=5, value=4, step=1)

# Reclassify if user changes threshold.
priority_gdf, hvi_join_status = classify_priority_areas(nta_metrics, hvi_raw, hvi_threshold=hvi_threshold)

filtered_gardens = filtered_by_borough(gardens, borough_choice)
filtered_priority = filtered_by_borough(priority_gdf, borough_choice)

st.sidebar.markdown("---")
st.sidebar.markdown("## Data status")
for label, status in [
    ("Gardens", greenthumb_status),
    ("NTA boundaries", nta_status),
    ("Population", population_status),
    ("HVI", hvi_join_status),
]:
    mock = status.get("mock", status.get("hvi_mock", "False"))
    icon = "⚠️" if str(mock) == "True" else "✅"
    st.sidebar.caption(f"{icon} {label}: {status.get('source', status.get('join_method', 'loaded'))}")

mock_flags = [
    greenthumb_status.get("mock") == "True",
    nta_status.get("mock") == "True",
    population_status.get("mock") == "True",
    hvi_join_status.get("hvi_mock") == "True",
]
if any(mock_flags):
    render_callout(
        "<strong>Demo caveat:</strong> One or more live data layers could not be joined or loaded and a mock fallback is being used. Replace mock population with a clean 2020 NTA population CSV before formal policy interpretation.",
        warning=True,
    )
else:
    render_callout(
        "<strong>Exploratory dashboard:</strong> This tool uses official NYC open data layers where available, but the analysis is designed for education and prioritization, not causal health claims.",
        warning=False,
    )

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Gardens Overview",
        "2. Access & Equity Map",
        "3. Climate Resilience Layer",
        "4. SDG Lens",
        "5. Youth Action & Recommendations",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Overview
# -----------------------------------------------------------------------------

with tab1:
    st.subheader("NYC Community Gardens Overview")

    total_gardens = len(filtered_gardens)
    borough_count = (
        filtered_gardens["borough"].nunique(dropna=True) if "borough" in filtered_gardens.columns else 0
    )
    nta_with_gardens = int((filtered_priority["garden_count"] > 0).sum())
    active_share = None
    if "status" in filtered_gardens.columns and total_gardens > 0:
        active_share = (
            filtered_gardens["status"].astype(str).str.contains("active|open", case=False, na=False).mean()
            * 100
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Total gardens", f"{total_gardens:,}", f"Filtered: {borough_choice}")
    with c2:
        render_kpi_card("Boroughs represented", f"{borough_count}", "Based on available borough field/spatial join")
    with c3:
        render_kpi_card("NTAs with ≥1 garden", f"{nta_with_gardens:,}", "Spatially joined to 2020 NTAs")
    with c4:
        render_kpi_card(
            "Active/open share",
            "N/A" if active_share is None else f"{active_share:.0f}%",
            "Only if status field is available",
        )

    render_callout(
        "Community gardens are distributed unevenly across NYC, creating an opportunity to examine access and equity."
    )

    st.pydeck_chart(make_garden_pydeck(filtered_gardens, buffer_distance), use_container_width=True)
    st.caption(
        f"The visual radius is set to {buffer_distance:.2f} miles. It is a communication aid, not a verified walking-network service area."
    )

    if "status" in filtered_gardens.columns and total_gardens > 0:
        status_counts = (
            filtered_gardens["status"].astype(str).replace("nan", "Unknown").value_counts().head(12)
        )
        if len(status_counts) > 1:
            fig_status = px.bar(
                status_counts.reset_index(),
                x="status",
                y="count",
                labels={"status": "Status", "count": "Garden count"},
                title="Status split, if available",
            )
            fig_status.update_layout(height=340, paper_bgcolor=PALETTE["cream"], plot_bgcolor="white")
            st.plotly_chart(fig_status, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Access and equity
# -----------------------------------------------------------------------------

with tab2:
    st.subheader("Access & Equity Map")

    render_callout(
        "<strong>Caveat:</strong> Population denominators, boundary years, and neighborhood definitions affect interpretation. Use this view to identify questions for deeper community validation, not to rank neighborhoods permanently.",
        warning=True,
    )

    map_metric = metric_view
    fig = make_choropleth(
        filtered_priority,
        map_metric,
        f"NTA choropleth by {friendly_metric_name(map_metric)}",
        color_scale="YlGn",
        overlay_points=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median access score", f"{city_median(filtered_priority['garden_access_score']):.2f}")
    with c2:
        st.metric("NTAs below city median", f"{int(filtered_priority['access_below_city_median'].sum()):,}")
    with c3:
        st.metric("Zero-garden NTAs", f"{int((filtered_priority['garden_count'] == 0).sum()):,}")

    st.markdown("### Bottom 10 NTAs by garden access score")
    bottom_cols = [
        "borough",
        "nta_code",
        "nta_name",
        "population",
        "garden_count",
        "garden_access_score",
        "garden_density_per_sq_mile",
    ]
    st.dataframe(
        filtered_priority[bottom_cols]
        .sort_values(["garden_access_score", "garden_count", "population"], ascending=[True, True, False])
        .head(10)
        .style.format(
            {
                "population": "{:,.0f}",
                "garden_access_score": "{:.2f}",
                "garden_density_per_sq_mile": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# Tab 3: Climate resilience
# -----------------------------------------------------------------------------

with tab3:
    st.subheader("Climate Resilience Layer")

    render_callout(
        "<strong>Exploratory Analysis/Priority Mapping Only:</strong> This matrix explores structural correlations, not direct causal health outcomes.",
        warning=True,
    )
    render_callout(
        "Neighborhoods with high heat vulnerability and low community garden access represent priority areas for youth-led sustainability education."
    )

    fig_hvi = make_choropleth(
        filtered_priority,
        "hvi_rank",
        "Heat Vulnerability Index with community garden overlay",
        color_scale="OrRd",
        overlay_points=filtered_gardens,
    )
    st.plotly_chart(fig_hvi, use_container_width=True)

    st.markdown("### Priority classification")
    p1, p2, p3 = st.columns(3)
    counts = filtered_priority["priority_class"].value_counts()
    with p1:
        st.metric("High priority", f"{int(counts.get('High priority', 0)):,}")
    with p2:
        st.metric("Medium priority", f"{int(counts.get('Medium priority', 0)):,}")
    with p3:
        st.metric("Lower priority", f"{int(counts.get('Lower priority', 0)):,}")

    priority_table_cols = [
        "priority_class",
        "borough",
        "nta_name",
        "hvi_rank",
        "garden_count",
        "garden_access_score",
        "population",
    ]
    st.dataframe(
        filtered_priority[priority_table_cols]
        .sort_values(["priority_sort", "garden_access_score", "hvi_rank"], ascending=[True, True, False])
        .head(15)
        .style.format({"garden_access_score": "{:.2f}", "population": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# Tab 4: SDG lens
# -----------------------------------------------------------------------------

with tab4:
    st.subheader("SDG Lens")
    st.markdown(
        "Community gardens are small spaces, but they connect multiple SDGs when they are treated as learning, resilience, and neighborhood partnership infrastructure."
    )

    sdgs = [
        {
            "sdg": "SDG 11",
            "title": "Sustainable Cities and Communities",
            "badge": "Inclusive public space",
            "text": "Gardens can improve neighborhood access to safe, welcoming green spaces and give residents a visible place to organize around local resilience.",
        },
        {
            "sdg": "SDG 13",
            "title": "Climate Action",
            "badge": "Heat awareness",
            "text": "Gardens can support climate adaptation education by helping students connect extreme heat, shade, soil, and local preparedness.",
        },
        {
            "sdg": "SDG 15",
            "title": "Life on Land",
            "badge": "Urban biodiversity",
            "text": "Gardens create small habitats for plants, pollinators, and soil life, making biodiversity visible in dense urban neighborhoods.",
        },
        {
            "sdg": "SDG 2",
            "title": "Zero Hunger",
            "badge": "Food education",
            "text": "Even when gardens do not solve food insecurity alone, they can teach food systems, nutrition, composting, and community food justice.",
        },
        {
            "sdg": "SDG 3",
            "title": "Good Health and Well-being",
            "badge": "Outdoor wellness",
            "text": "Gardens can support outdoor learning, social connection, stress reduction, and youth-led community health conversations.",
        },
    ]

    row1 = st.columns(3)
    row2 = st.columns(2)
    slots = row1 + row2
    for slot, item in zip(slots, sdgs):
        with slot:
            st.markdown(
                f"""
                <div class='sdg-card'>
                    <span class='badge'>{item['sdg']}</span>
                    <h4 style='color:{PALETTE['forest']}; margin-top:.55rem'>{item['title']}</h4>
                    <strong style='color:{PALETTE['terracotta']}'>{item['badge']}</strong>
                    <p style='margin-top:.55rem'>{item['text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Tab 5: Youth action and recommendations
# -----------------------------------------------------------------------------

with tab5:
    st.subheader("Youth Action & Recommendations")

    priority_for_action = filtered_priority.sort_values(
        ["priority_sort", "garden_access_score", "hvi_rank"], ascending=[True, True, False]
    ).copy()
    top_priority = priority_for_action.head(5)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            """
            <div class='action-card'>
                <h4>1. What students can learn</h4>
                <ul>
                    <li>How heat vulnerability, green space, and public health are connected.</li>
                    <li>How to read maps critically and ask what data may be missing.</li>
                    <li>How community-led spaces can become climate education infrastructure.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        neighborhoods = "".join(
            f"<li><strong>{row.nta_name}</strong> — {row.borough}, HVI {row.hvi_rank:.0f}, access {row.garden_access_score:.2f}</li>"
            for _, row in top_priority.iterrows()
        )
        if not neighborhoods:
            neighborhoods = "<li>No priority neighborhoods available under the current filter.</li>"
        st.markdown(
            f"""
            <div class='action-card'>
                <h4>2. Priority neighborhoods to adopt</h4>
                <ul>{neighborhoods}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            """
            <div class='action-card'>
                <h4>3. SDG action ideas</h4>
                <ul>
                    <li>Create a student-made heat and garden awareness map.</li>
                    <li>Design a pollinator, composting, or shade-learning station.</li>
                    <li>Host a one-day SDG garden walk connecting SDG 11, 13, 15, 2, and 3.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            """
            <div class='action-card'>
                <h4>4. Community partnership ideas</h4>
                <ul>
                    <li>Partner with GreenThumb garden groups for local storytelling.</li>
                    <li>Invite public health, urban planning, and school sustainability mentors.</li>
                    <li>Share findings with neighborhood councils or youth climate clubs.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Download priority neighborhoods")
    download_df = priority_download_df(priority_for_action)
    st.download_button(
        label="Download priority neighborhoods CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="nyc_garden_priority_neighborhoods.csv",
        mime="text/csv",
    )

    presenter_script = f"""
Hello, my dashboard asks one question: How can NYC community gardens support sustainable, equitable, and climate-resilient cities?
I treat community gardens as small-scale urban resilience infrastructure. First, the dashboard maps where gardens are located across New York City. Then it connects those garden locations to Neighborhood Tabulation Areas, population denominators, and heat vulnerability.
The key insight is not just how many gardens exist, but where access is lower and climate risk is higher. In this dashboard, high-priority neighborhoods are places where heat vulnerability is high and garden access is below the city median. That makes them strong candidates for youth-led sustainability education, garden partnerships, and SDG action.
This is exploratory priority mapping, not a causal health study. But it helps students ask better civic questions: Who has access to green space? Which neighborhoods face greater climate stress? And how can young people turn local gardens into learning spaces for SDG 11, climate action, biodiversity, food education, and community well-being?
""".strip()

    st.markdown("### 60-second presenter script")
    st.text_area("Presenter script", value=presenter_script, height=230)

# -----------------------------------------------------------------------------
# Footer: data caveats and sources
# -----------------------------------------------------------------------------

st.markdown("---")
with st.expander("Data caveats and source notes", expanded=False):
    st.markdown(
        f"""
        **Primary official NYC data IDs used in this app**
        - GreenThumb Garden Info: `{GREENTHUMB_ID}`
        - 2020 Neighborhood Tabulation Areas: `{NTA_ID}`
        - Heat Vulnerability Index Rankings: `{HVI_ID}`

        **Key caveats**
        1. This dashboard is exploratory and educational. It does not estimate causal effects of gardens on heat illness, health outcomes, or food security.
        2. Population denominators strongly affect access scores. For production use, replace the placeholder with a clean 2020 NTA population file at `data/nta_population.csv`.
        3. HVI geography and NTA geography may not share a perfect join key. The code attempts code/name joins and clearly falls back to mock HVI when the match is weak.
        4. Garden access is approximated using point-in-polygon counts, not walking-network distance, garden opening hours, accessibility, capacity, or program quality.
        5. The visual buffer radius is for communication only. A formal access analysis should use pedestrian network travel time and entrance points.
        6. NTA boundaries are statistical geographies; they may not match residents' lived neighborhood identities.
        """
    )
    st.json(
        {
            "greenthumb_status": greenthumb_status,
            "garden_clean_status": garden_clean_status,
            "nta_status": nta_status,
            "population_status": population_status,
            "metrics_status": metrics_status,
            "hvi_load_status": hvi_load_status,
            "hvi_join_status": hvi_join_status,
        }
    )
