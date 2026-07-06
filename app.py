"""
NYC Community Gardens for Equitable Climate Resilience
Streamlit dashboard designed for Streamlit Community Cloud deployment.

This version intentionally avoids GeoPandas/Fiona/GDAL to prevent common cloud
installation failures. It still performs true point-in-polygon spatial assignment
using Shapely and stores geometries inside pandas DataFrames.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Primary NYC Open Data source IDs:
    GreenThumb Garden Info: p78i-pat6
    2020 Neighborhood Tabulation Areas: 9nt8-h7nd
    Heat Vulnerability Index Rankings: 4mhf-duep

Optional local population file:
    data/nta_population.csv with columns: nta_code, nta_name, population
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st
from shapely import wkt
from shapely.geometry import Point, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

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
# Constants and visual system
# -----------------------------------------------------------------------------

GREENTHUMB_ID = "p78i-pat6"
NTA_ID = "9nt8-h7nd"
HVI_ID = "4mhf-duep"

SODA_JSON = "https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit=50000"
SODA_GEOJSON = "https://data.cityofnewyork.us/api/views/{dataset_id}/rows.geojson?accessType=DOWNLOAD"
SODA_GEOSPATIAL_EXPORT = (
    "https://data.cityofnewyork.us/api/geospatial/{dataset_id}?method=export&format=GeoJSON"
)

NYC_CENTER_LAT = 40.7128
NYC_CENTER_LON = -74.0060
WGS84_NOTE = "Geometries are handled in longitude/latitude coordinates for web mapping."

BOROUGH_ORDER = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
BOROUGH_ALIASES = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
    "mn": "Manhattan",
    "bx": "Bronx",
    "bk": "Brooklyn",
    "qn": "Queens",
    "si": "Staten Island",
    "manhattan": "Manhattan",
    "bronx": "Bronx",
    "brooklyn": "Brooklyn",
    "queens": "Queens",
    "staten island": "Staten Island",
    "staten_island": "Staten Island",
}

PALETTE = {
    "forest": "#12372A",
    "sage": "#9CAF88",
    "cream": "#F6F2E8",
    "terracotta": "#C76F4A",
    "mint": "#DDE8D1",
    "ink": "#1E2A24",
    "muted": "#66756D",
    "white": "#FFFFFF",
    "gold": "#D9A441",
    "red": "#B85042",
}

PRIORITY_COLORS = {
    "High priority": "#B85042",
    "Medium priority": "#D9A441",
    "Lower priority": "#9CAF88",
}

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------

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
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 850;
            color: {PALETTE['forest']};
            margin-bottom: 0.2rem;
        }}
        .guiding-question {{
            font-size: 1.24rem;
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
        .action-card, .sdg-card {{
            background: #FFFFFF;
            border: 1px solid #E6E1D5;
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 12px rgba(18, 55, 42, 0.07);
            min-height: 176px;
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
# Generic helpers
# -----------------------------------------------------------------------------


def normalize_col_name(col: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(col).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for candidate in candidates:
        normalized = normalize_col_name(candidate)
        if normalized in cols:
            return normalized
    return None


def normalize_text_key(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def standardize_borough(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    key = normalize_text_key(raw).replace(" ", "_")
    if key in BOROUGH_ALIASES:
        return BOROUGH_ALIASES[key]
    key2 = normalize_text_key(raw)
    return BOROUGH_ALIASES.get(key2, raw.title())


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def try_request_json(url: str, timeout: int = 25) -> object:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def geometry_to_jsonable(geom: BaseGeometry) -> Dict:
    return json.loads(json.dumps(mapping(geom)))


def df_to_feature_collection(
    df: pd.DataFrame,
    id_col: str = "nta_code",
    property_cols: Optional[List[str]] = None,
) -> Dict:
    features = []
    if property_cols is None:
        property_cols = [c for c in df.columns if c != "geometry"]
    for _, row in df.iterrows():
        geom = row.get("geometry")
        if not isinstance(geom, BaseGeometry) or geom.is_empty:
            continue
        props = {}
        for col in property_cols:
            value = row.get(col)
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif pd.isna(value) if not isinstance(value, (list, dict, BaseGeometry)) else False:
                value = None
            props[col] = value
        feature = {
            "type": "Feature",
            "id": str(row.get(id_col, len(features))),
            "properties": props,
            "geometry": geometry_to_jsonable(geom),
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def parse_geojson_features(geojson_obj: Dict) -> pd.DataFrame:
    records = []
    for feature in geojson_obj.get("features", []):
        props = feature.get("properties") or {}
        geom_obj = feature.get("geometry")
        try:
            geom = shape(geom_obj) if geom_obj else None
        except Exception:
            geom = None
        props = {normalize_col_name(k): v for k, v in props.items()}
        props["geometry"] = geom
        records.append(props)
    return pd.DataFrame(records)


def extract_geometry_from_any_column(df: pd.DataFrame) -> pd.Series:
    """Parse Socrata-style geometry/WKT columns when present."""
    candidates = ["the_geom", "geometry", "geom", "multipolygon", "polygon", "point"]
    geom_col = first_existing(df, candidates)
    if geom_col is None:
        return pd.Series([None] * len(df), index=df.index)

    parsed = []
    for value in df[geom_col]:
        geom = None
        try:
            if isinstance(value, BaseGeometry):
                geom = value
            elif isinstance(value, dict):
                # Socrata sometimes returns {"type":"Point", "coordinates":[lon,lat]}
                geom = shape(value)
            elif isinstance(value, str) and value.strip():
                val = value.strip()
                if val.startswith("{"):
                    geom = shape(json.loads(val))
                else:
                    geom = wkt.loads(val)
        except Exception:
            geom = None
        parsed.append(geom)
    return pd.Series(parsed, index=df.index)


def polygon_area_sq_mi(geom: BaseGeometry) -> float:
    """
    Approximate area in square miles without pyproj.

    For a civic-tech educational dashboard, this is sufficient for comparative
    density screening. Replace with a projected CRS workflow for official analysis.
    """
    if not isinstance(geom, BaseGeometry) or geom.is_empty:
        return np.nan

    def ring_area(coords) -> float:
        if len(coords) < 3:
            return 0.0
        lat0 = np.mean([lat for lon, lat in coords])
        cos_lat = math.cos(math.radians(lat0))
        pts = [((lon - NYC_CENTER_LON) * 69.172 * cos_lat, (lat - NYC_CENTER_LAT) * 69.0) for lon, lat in coords]
        area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def polygon_area(poly) -> float:
        area = ring_area(list(poly.exterior.coords))
        for interior in poly.interiors:
            area -= ring_area(list(interior.coords))
        return max(area, 0.0)

    try:
        if geom.geom_type == "Polygon":
            return polygon_area(geom)
        if geom.geom_type == "MultiPolygon":
            return sum(polygon_area(poly) for poly in geom.geoms)
    except Exception:
        return np.nan
    return np.nan


def format_int(value: object) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "—"


def format_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"

# -----------------------------------------------------------------------------
# Mock fallbacks
# -----------------------------------------------------------------------------


def create_mock_nta_data() -> pd.DataFrame:
    rows = []
    borough_specs = [
        ("Bronx", -73.93, 40.80, 3, 3),
        ("Manhattan", -74.02, 40.70, 2, 4),
        ("Brooklyn", -74.02, 40.58, 4, 3),
        ("Queens", -73.90, 40.58, 4, 3),
        ("Staten Island", -74.22, 40.50, 2, 2),
    ]
    idx = 1
    for borough, lon0, lat0, nx, ny in borough_specs:
        for ix in range(nx):
            for iy in range(ny):
                lon = lon0 + ix * 0.045
                lat = lat0 + iy * 0.045
                code_prefix = {"Bronx": "BX", "Manhattan": "MN", "Brooklyn": "BK", "Queens": "QN", "Staten Island": "SI"}[borough]
                rows.append(
                    {
                        "nta_code": f"{code_prefix}{idx:03d}",
                        "nta_name": f"Mock {borough} NTA {idx}",
                        "borough": borough,
                        "geometry": box(lon, lat, lon + 0.037, lat + 0.037),
                        "is_mock": True,
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def create_mock_garden_data() -> pd.DataFrame:
    rng = random.Random(42)
    rows = []
    centers = {
        "Bronx": (-73.90, 40.84),
        "Brooklyn": (-73.95, 40.67),
        "Manhattan": (-73.98, 40.77),
        "Queens": (-73.82, 40.71),
        "Staten Island": (-74.15, 40.58),
    }
    counts = {"Bronx": 36, "Brooklyn": 58, "Manhattan": 38, "Queens": 30, "Staten Island": 8}
    garden_id = 1
    for borough, (lon0, lat0) in centers.items():
        for _ in range(counts[borough]):
            lon = lon0 + rng.uniform(-0.08, 0.08)
            lat = lat0 + rng.uniform(-0.06, 0.06)
            rows.append(
                {
                    "garden_id": f"MOCK-G{garden_id:04d}",
                    "garden_name": f"Mock Community Garden {garden_id}",
                    "borough": borough,
                    "status": rng.choice(["Active", "Active", "Active", "Pending"]),
                    "geometry": Point(lon, lat),
                    "lat": lat,
                    "lon": lon,
                    "is_mock": True,
                }
            )
            garden_id += 1
    return pd.DataFrame(rows)


def create_mock_population_data(nta_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(7)
    rows = []
    for _, row in nta_df.iterrows():
        borough = row.get("borough", "Unknown")
        base = {
            "Manhattan": 52000,
            "Brooklyn": 43000,
            "Bronx": 39000,
            "Queens": 38000,
            "Staten Island": 28000,
        }.get(borough, 35000)
        rows.append(
            {
                "nta_code": row.get("nta_code"),
                "nta_name": row.get("nta_name"),
                "population": int(base * rng.uniform(0.55, 1.35)),
                "is_mock_population": True,
            }
        )
    return pd.DataFrame(rows)


def create_mock_hvi_data(nta_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(13)
    rows = []
    for _, row in nta_df.iterrows():
        borough = row.get("borough", "Unknown")
        bias = {"Bronx": 1.0, "Brooklyn": 0.45, "Manhattan": -0.15, "Queens": 0.25, "Staten Island": -0.25}.get(borough, 0)
        score = int(np.clip(round(rng.uniform(1, 5) + bias), 1, 5))
        rows.append(
            {
                "nta_code": row.get("nta_code"),
                "nta_name": row.get("nta_name"),
                "hvi_rank": score,
                "is_mock_hvi": True,
            }
        )
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_greenthumb_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load GreenThumb garden records from NYC Open Data.

    The function first tries GeoJSON because it preserves geometry. If that fails,
    it tries the Socrata JSON API. If both fail, it returns mock points and sets
    a clear status flag for the UI.
    """
    status = {"source": "NYC Open Data GreenThumb Garden Info", "is_mock": "False", "message": ""}

    for url in [
        SODA_GEOJSON.format(dataset_id=GREENTHUMB_ID),
        SODA_GEOSPATIAL_EXPORT.format(dataset_id=GREENTHUMB_ID),
    ]:
        try:
            geojson_obj = try_request_json(url)
            df = parse_geojson_features(geojson_obj)
            if len(df) > 0:
                status["message"] = "Loaded GreenThumb records from NYC Open Data GeoJSON."
                return normalize_columns(df), status
        except Exception as exc:
            status["message"] = f"GeoJSON load failed: {exc}"

    try:
        data = try_request_json(SODA_JSON.format(dataset_id=GREENTHUMB_ID))
        df = normalize_columns(pd.DataFrame(data))
        if len(df) > 0:
            status["message"] = "Loaded GreenThumb records from NYC Open Data JSON."
            return df, status
    except Exception as exc:
        status["message"] = f"JSON load failed: {exc}"

    status["source"] = "Mock GreenThumb fallback"
    status["is_mock"] = "True"
    status["message"] = "Live GreenThumb loading failed. Using mock garden points."
    return create_mock_garden_data(), status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_nta_boundaries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load 2020 Neighborhood Tabulation Area boundaries from NYC Open Data."""
    status = {"source": "NYC Open Data 2020 NTA Boundaries", "is_mock": "False", "message": ""}
    for url in [
        SODA_GEOSPATIAL_EXPORT.format(dataset_id=NTA_ID),
        SODA_GEOJSON.format(dataset_id=NTA_ID),
    ]:
        try:
            geojson_obj = try_request_json(url, timeout=35)
            df = parse_geojson_features(geojson_obj)
            if len(df) > 0 and "geometry" in df.columns:
                df = normalize_columns(df)
                code_col = first_existing(df, ["nta2020", "ntacode", "nta_code", "geoid", "geoid20"])
                name_col = first_existing(df, ["ntaname", "nta_name", "ntaname2020", "name"])
                boro_col = first_existing(df, ["boroname", "borough", "boro_name", "borocode", "boro_code"])

                if code_col is None:
                    df["nta_code"] = [f"NTA_{i:03d}" for i in range(len(df))]
                else:
                    df["nta_code"] = df[code_col].astype(str)

                if name_col is None:
                    df["nta_name"] = df["nta_code"]
                else:
                    df["nta_name"] = df[name_col].astype(str)

                if boro_col is None:
                    df["borough"] = "Unknown"
                else:
                    df["borough"] = df[boro_col].map(standardize_borough)

                df = df[df["geometry"].apply(lambda g: isinstance(g, BaseGeometry) and not g.is_empty)].copy()
                df["area_sq_mi"] = df["geometry"].apply(polygon_area_sq_mi)
                df["is_mock"] = False
                status["message"] = "Loaded 2020 NTA boundaries from NYC Open Data."
                return df[["nta_code", "nta_name", "borough", "geometry", "area_sq_mi", "is_mock"]], status
        except Exception as exc:
            status["message"] = f"Boundary load failed: {exc}"

    status["source"] = "Mock NTA fallback"
    status["is_mock"] = "True"
    status["message"] = "Live NTA boundary loading failed. Using mock polygons."
    mock = create_mock_nta_data()
    mock["area_sq_mi"] = mock["geometry"].apply(polygon_area_sq_mi)
    return mock, status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_hvi_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load Heat Vulnerability Index rankings from NYC Open Data."""
    status = {"source": "NYC Open Data Heat Vulnerability Index Rankings", "is_mock": "False", "message": ""}
    try:
        data = try_request_json(SODA_JSON.format(dataset_id=HVI_ID), timeout=25)
        df = normalize_columns(pd.DataFrame(data))
        if len(df) > 0:
            status["message"] = "Loaded HVI records from NYC Open Data JSON."
            return df, status
    except Exception as exc:
        status["message"] = f"HVI JSON load failed: {exc}"

    # Actual mock HVI needs NTA rows, so classify_priority_areas() can generate it.
    status["source"] = "Mock HVI fallback pending"
    status["is_mock"] = "True"
    status["message"] = "Live HVI loading failed. Mock HVI will be generated after NTA load."
    return pd.DataFrame(), status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_population_data(nta_signature: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load NTA-level population data.

    Recommended production replacement:
        Add data/nta_population.csv to the GitHub repo with columns:
            nta_code, nta_name, population

    The user requested NYC Population FactFinder or a clean local CSV placeholder;
    Population FactFinder is commonly used as a web tool, so this dashboard uses
    a local CSV when present and a clearly marked mock fallback otherwise.
    """
    status = {"source": "Local data/nta_population.csv or mock fallback", "is_mock": "False", "message": ""}
    local_paths = ["data/nta_population.csv", "nta_population.csv"]
    for path in local_paths:
        try:
            df = normalize_columns(pd.read_csv(path))
            pop_col = first_existing(df, ["population", "pop", "total_population", "pop_total"])
            code_col = first_existing(df, ["nta_code", "nta2020", "ntacode", "geoid"])
            name_col = first_existing(df, ["nta_name", "ntaname", "ntaname2020", "name"])
            if pop_col is not None and (code_col is not None or name_col is not None):
                out = pd.DataFrame()
                if code_col is not None:
                    out["nta_code"] = df[code_col].astype(str)
                if name_col is not None:
                    out["nta_name"] = df[name_col].astype(str)
                out["population"] = pd.to_numeric(df[pop_col], errors="coerce")
                out = out.dropna(subset=["population"])
                out["is_mock_population"] = False
                status["source"] = path
                status["message"] = f"Loaded population data from {path}."
                return out, status
        except FileNotFoundError:
            continue
        except Exception as exc:
            status["message"] = f"Local population load failed from {path}: {exc}"

    status["source"] = "Mock population fallback"
    status["is_mock"] = "True"
    status["message"] = "No local NTA population CSV found. Using mock population denominators."
    # nta_signature is used only to keep st.cache_data dependency explicit.
    _ = nta_signature
    return pd.DataFrame(), status

# -----------------------------------------------------------------------------
# Cleaning and analysis functions
# -----------------------------------------------------------------------------


def clean_garden_points(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert raw garden records into point locations.

    Priority:
    1. Existing geometry column, including polygons converted to representative points.
    2. Latitude/longitude columns.
    3. WKT/GeoJSON geometry columns.
    4. Mock sample fallback with a visible caveat.
    """
    status = {"is_mock": "False", "message": "Garden locations cleaned from available geometry/coordinates."}
    df = normalize_columns(raw_data)

    # Garden name, id, borough, status standardization
    id_col = first_existing(df, ["garden_id", "propid", "property_id", "id", "objectid"])
    name_col = first_existing(df, ["garden_name", "name", "garden", "signname", "propertyname"])
    borough_col = first_existing(df, ["borough", "boroname", "boro", "boro_name", "borocode", "boro_code"])
    status_col = first_existing(df, ["status", "garden_status", "active", "status_of_garden"])

    if id_col is None:
        df["garden_id"] = [f"GARDEN_{i:05d}" for i in range(len(df))]
    else:
        df["garden_id"] = df[id_col].astype(str)

    if name_col is None:
        df["garden_name"] = df["garden_id"]
    else:
        df["garden_name"] = df[name_col].fillna(df["garden_id"]).astype(str)

    if borough_col is None:
        df["borough"] = "Unknown"
    else:
        df["borough"] = df[borough_col].map(standardize_borough)

    if status_col is None:
        df["status"] = "Unknown"
    else:
        df["status"] = df[status_col].fillna("Unknown").astype(str).str.title()

    geometry_series = None
    if "geometry" in df.columns and df["geometry"].apply(lambda g: isinstance(g, BaseGeometry)).any():
        geometry_series = df["geometry"]
    else:
        geometry_series = extract_geometry_from_any_column(df)

    point_geoms = []
    if geometry_series is not None and geometry_series.apply(lambda g: isinstance(g, BaseGeometry)).any():
        for geom in geometry_series:
            if not isinstance(geom, BaseGeometry) or geom.is_empty:
                point_geoms.append(None)
            elif geom.geom_type == "Point":
                point_geoms.append(geom)
            else:
                try:
                    point_geoms.append(geom.representative_point())
                except Exception:
                    point_geoms.append(geom.centroid)
    else:
        point_geoms = [None] * len(df)

    # Latitude/longitude fallback or override for rows with missing geometry
    lat_col = first_existing(df, ["latitude", "lat", "y", "garden_latitude"])
    lon_col = first_existing(df, ["longitude", "lon", "lng", "x", "garden_longitude"])
    if lat_col is not None and lon_col is not None:
        for i, (lat_val, lon_val) in enumerate(zip(df[lat_col], df[lon_col])):
            if point_geoms[i] is None:
                lat = safe_float(lat_val)
                lon = safe_float(lon_val)
                if lat is not None and lon is not None and -75.0 <= lon <= -72.5 and 40.0 <= lat <= 41.2:
                    point_geoms[i] = Point(lon, lat)

    df["geometry"] = point_geoms
    df = df[df["geometry"].apply(lambda g: isinstance(g, BaseGeometry) and not g.is_empty)].copy()

    if len(df) == 0:
        status["is_mock"] = "True"
        status["message"] = "Latitude/longitude and geometry were missing or unreadable. Using mock garden points."
        return create_mock_garden_data(), status

    df["lon"] = df["geometry"].apply(lambda g: g.x)
    df["lat"] = df["geometry"].apply(lambda g: g.y)
    df["is_mock"] = df.get("is_mock", False)
    return df[["garden_id", "garden_name", "borough", "status", "geometry", "lat", "lon", "is_mock"]].copy(), status


def compute_nta_metrics(
    garden_points: pd.DataFrame,
    nta_boundaries: pd.DataFrame,
    population_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Spatially assign gardens to NTAs and compute access/density metrics."""
    status = {"message": "Computed NTA garden metrics with point-in-polygon assignment."}
    nta = nta_boundaries.copy()
    gardens = garden_points.copy()

    if len(population_df) == 0:
        population_df = create_mock_population_data(nta)

    population_df = normalize_columns(population_df)
    pop_col = first_existing(population_df, ["population", "pop", "total_population", "pop_total"])
    code_col = first_existing(population_df, ["nta_code", "nta2020", "ntacode", "geoid"])
    name_col = first_existing(population_df, ["nta_name", "ntaname", "ntaname2020", "name"])

    if pop_col is None:
        population_df = create_mock_population_data(nta)
        pop_col = "population"
        code_col = "nta_code"
        name_col = "nta_name"

    pop = pd.DataFrame()
    if code_col is not None:
        pop["nta_code"] = population_df[code_col].astype(str)
    if name_col is not None:
        pop["nta_name_key"] = population_df[name_col].map(normalize_text_key)
    pop["population"] = pd.to_numeric(population_df[pop_col], errors="coerce")
    pop = pop.dropna(subset=["population"])

    nta["nta_name_key"] = nta["nta_name"].map(normalize_text_key)
    if "nta_code" in pop.columns:
        nta = nta.merge(pop[["nta_code", "population"]], on="nta_code", how="left")
    else:
        nta["population"] = np.nan

    if nta["population"].isna().any() and "nta_name_key" in pop.columns:
        nta = nta.merge(
            pop[["nta_name_key", "population"]].rename(columns={"population": "population_by_name"}),
            on="nta_name_key",
            how="left",
        )
        nta["population"] = nta["population"].fillna(nta["population_by_name"])
        nta = nta.drop(columns=["population_by_name"])

    if nta["population"].isna().any():
        mock_pop = create_mock_population_data(nta)
        nta = nta.drop(columns=["population"], errors="ignore").merge(
            mock_pop[["nta_code", "population"]], on="nta_code", how="left"
        )
        status["message"] += " Population was missing for some/all NTAs, so mock denominators were used."

    if "area_sq_mi" not in nta.columns or nta["area_sq_mi"].isna().all():
        nta["area_sq_mi"] = nta["geometry"].apply(polygon_area_sq_mi)

    # Prepare polygons once. NTA count is small enough for transparent nested loops.
    prepared_polys = []
    for idx, row in nta.iterrows():
        geom = row.get("geometry")
        if isinstance(geom, BaseGeometry) and not geom.is_empty:
            prepared_polys.append((idx, geom, prep(geom)))

    assigned_records = []
    for _, garden in gardens.iterrows():
        point = garden.get("geometry")
        assigned_idx = None
        if isinstance(point, BaseGeometry) and not point.is_empty:
            for idx, geom, prepared in prepared_polys:
                try:
                    if prepared.contains(point) or geom.touches(point):
                        assigned_idx = idx
                        break
                except Exception:
                    continue
        record = garden.to_dict()
        if assigned_idx is not None:
            record["nta_code"] = nta.loc[assigned_idx, "nta_code"]
            record["nta_name"] = nta.loc[assigned_idx, "nta_name"]
            record["nta_borough"] = nta.loc[assigned_idx, "borough"]
            if record.get("borough") in [None, "Unknown", ""]:
                record["borough"] = nta.loc[assigned_idx, "borough"]
        else:
            record["nta_code"] = None
            record["nta_name"] = None
            record["nta_borough"] = None
        assigned_records.append(record)

    joined = pd.DataFrame(assigned_records)
    counts = joined.dropna(subset=["nta_code"]).groupby("nta_code").size().reset_index(name="garden_count")
    nta = nta.merge(counts, on="nta_code", how="left")
    nta["garden_count"] = nta["garden_count"].fillna(0).astype(int)
    nta["garden_access_score"] = np.where(
        nta["population"] > 0,
        nta["garden_count"] / nta["population"] * 10000,
        np.nan,
    )
    nta["garden_density_sqmi"] = np.where(
        nta["area_sq_mi"] > 0,
        nta["garden_count"] / nta["area_sq_mi"],
        np.nan,
    )
    nta["access_label"] = nta["garden_access_score"].apply(lambda x: f"{x:.2f} per 10k" if pd.notna(x) else "N/A")
    return nta, joined, status


def classify_priority_areas(
    nta_metrics: pd.DataFrame,
    hvi_df: pd.DataFrame,
    hvi_threshold: int = 4,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Join HVI rankings and classify high/medium/lower priority areas."""
    status = {"message": "Joined HVI data and classified priority areas.", "is_mock_hvi": "False"}
    nta = nta_metrics.copy()

    if len(hvi_df) == 0:
        hvi_df = create_mock_hvi_data(nta)
        status["is_mock_hvi"] = "True"
        status["message"] = "Live HVI unavailable. Using mock HVI rankings."

    hvi = normalize_columns(hvi_df)
    hvi_col = first_existing(
        hvi,
        [
            "hvi_rank",
            "heat_vulnerability_index",
            "heat_vulnerability_index_rank",
            "hvi",
            "rank",
            "score",
            "hvi_ranking",
        ],
    )
    code_col = first_existing(hvi, ["nta_code", "nta2020", "ntacode", "geoid", "nta"])
    name_col = first_existing(
        hvi,
        ["nta_name", "ntaname", "ntaname2020", "neighborhood", "neighborhood_name", "area_name", "name"],
    )

    if hvi_col is None:
        hvi = create_mock_hvi_data(nta)
        hvi_col = "hvi_rank"
        code_col = "nta_code"
        name_col = "nta_name"
        status["is_mock_hvi"] = "True"
        status["message"] = "HVI rank column not detected. Using mock HVI rankings."

    clean_hvi = pd.DataFrame()
    clean_hvi["hvi_rank"] = pd.to_numeric(hvi[hvi_col], errors="coerce")
    if code_col is not None:
        clean_hvi["nta_code"] = hvi[code_col].astype(str)
    if name_col is not None:
        clean_hvi["nta_name_key"] = hvi[name_col].map(normalize_text_key)

    nta["nta_name_key"] = nta["nta_name"].map(normalize_text_key)

    joined = nta.copy()
    if "nta_code" in clean_hvi.columns:
        joined = joined.merge(clean_hvi[["nta_code", "hvi_rank"]].dropna(), on="nta_code", how="left")
    else:
        joined["hvi_rank"] = np.nan

    if joined["hvi_rank"].isna().all() and "nta_name_key" in clean_hvi.columns:
        joined = joined.drop(columns=["hvi_rank"], errors="ignore").merge(
            clean_hvi[["nta_name_key", "hvi_rank"]].dropna(), on="nta_name_key", how="left"
        )

    if joined["hvi_rank"].isna().all():
        mock_hvi = create_mock_hvi_data(joined)
        joined = joined.drop(columns=["hvi_rank"], errors="ignore").merge(
            mock_hvi[["nta_code", "hvi_rank"]], on="nta_code", how="left"
        )
        status["is_mock_hvi"] = "True"
        status["message"] = "Could not match HVI to NTA geography. Using mock HVI rankings."

    median_access = joined["garden_access_score"].median(skipna=True)
    if pd.isna(median_access):
        median_access = 0
    joined["city_median_access_score"] = median_access

    def priority(row) -> str:
        hvi = row.get("hvi_rank")
        access = row.get("garden_access_score")
        if pd.isna(hvi) or pd.isna(access):
            return "Lower priority"
        if hvi >= hvi_threshold and access < median_access:
            return "High priority"
        if hvi >= 3 and access < median_access:
            return "Medium priority"
        return "Lower priority"

    joined["priority_class"] = joined.apply(priority, axis=1)
    joined["priority_sort"] = joined["priority_class"].map({"High priority": 1, "Medium priority": 2, "Lower priority": 3})
    return joined, status

# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def make_kpi_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="subtle">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_garden_pydeck(gardens: pd.DataFrame, buffer_miles: float = 0.25) -> pdk.Deck:
    df = gardens.dropna(subset=["lat", "lon"]).copy()
    df["garden_label"] = df["garden_name"].astype(str)
    df["buffer_radius_m"] = buffer_miles * 1609.34

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_radius=55,
            get_fill_color=[18, 55, 42, 190],
            pickable=True,
        )
    ]
    if buffer_miles > 0:
        layers.insert(
            0,
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_radius="buffer_radius_m",
                get_fill_color=[156, 175, 136, 36],
                pickable=False,
            ),
        )

    return pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=NYC_CENTER_LAT, longitude=NYC_CENTER_LON, zoom=10, pitch=0),
        layers=layers,
        tooltip={"html": "<b>{garden_label}</b><br/>{borough}<br/>{status}"},
    )


def make_choropleth(
    df: pd.DataFrame,
    metric: str,
    title: str,
    color_scale: str = "YlGn",
    overlay_points: Optional[pd.DataFrame] = None,
) -> object:
    plot_df = df.copy()
    plot_df["_id"] = plot_df["nta_code"].astype(str)
    geojson = df_to_feature_collection(plot_df, id_col="_id")
    metric_label = {
        "garden_count": "Garden Count",
        "garden_access_score": "Gardens per 10,000 Residents",
        "garden_density_sqmi": "Gardens per Square Mile",
        "hvi_rank": "Heat Vulnerability Index Rank",
        "priority_numeric": "Priority Score",
    }.get(metric, metric)

    fig = px.choropleth_mapbox(
        plot_df,
        geojson=geojson,
        locations="_id",
        featureidkey="id",
        color=metric,
        hover_name="nta_name",
        hover_data={
            "borough": True,
            "garden_count": ":,.0f" if "garden_count" in plot_df else False,
            "garden_access_score": ":.2f" if "garden_access_score" in plot_df else False,
            "garden_density_sqmi": ":.2f" if "garden_density_sqmi" in plot_df else False,
            "hvi_rank": ":.0f" if "hvi_rank" in plot_df else False,
            "priority_class": True if "priority_class" in plot_df else False,
            "_id": False,
        },
        color_continuous_scale=color_scale,
        opacity=0.72,
        mapbox_style="carto-positron",
        center={"lat": NYC_CENTER_LAT, "lon": NYC_CENTER_LON},
        zoom=9.45,
        labels={metric: metric_label},
        title=title,
    )
    fig.update_layout(
        height=610,
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        paper_bgcolor=PALETTE["cream"],
        plot_bgcolor=PALETTE["cream"],
        font={"color": PALETTE["ink"]},
    )

    if overlay_points is not None and len(overlay_points) > 0:
        pts = overlay_points.dropna(subset=["lat", "lon"]).copy()
        fig.add_scattermapbox(
            lat=pts["lat"],
            lon=pts["lon"],
            mode="markers",
            marker={"size": 6, "color": PALETTE["forest"], "opacity": 0.72},
            text=pts["garden_name"],
            name="Community gardens",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    return fig


def priority_download_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "nta_code",
        "nta_name",
        "borough",
        "population",
        "garden_count",
        "garden_access_score",
        "garden_density_sqmi",
        "hvi_rank",
        "priority_class",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for col in ["garden_access_score", "garden_density_sqmi"]:
        if col in out.columns:
            out[col] = out[col].round(3)
    return out.sort_values(["priority_class", "hvi_rank", "garden_access_score"], ascending=[True, False, True])

# -----------------------------------------------------------------------------
# App data pipeline
# -----------------------------------------------------------------------------

with st.spinner("Loading NYC community garden and neighborhood data..."):
    nta_raw, nta_status = load_nta_boundaries()
    gardens_raw, gardens_status = load_greenthumb_data()
    hvi_raw, hvi_status = load_hvi_data()
    pop_raw, pop_status = load_population_data(str(len(nta_raw)))
    gardens_clean, clean_status = clean_garden_points(gardens_raw)
    nta_metrics, garden_joined, metric_status = compute_nta_metrics(gardens_clean, nta_raw, pop_raw)

# Sidebar filters need computed data.
st.sidebar.markdown("## Global filters")
borough_options = ["All NYC"] + [b for b in BOROUGH_ORDER if b in set(nta_metrics["borough"])]
selected_borough = st.sidebar.selectbox("Borough", borough_options, index=0)
metric_view = st.sidebar.selectbox(
    "Metric view",
    ["Garden Access Score", "Garden Count", "Garden Density"],
    index=0,
)
buffer_distance = st.sidebar.slider("Garden buffer distance", 0.0, 1.0, 0.25, 0.05, help="Visual reference only, not a network walk-time analysis.")
hvi_threshold = st.sidebar.slider("High HVI threshold", 3, 5, 4, 1)

priority_df, priority_status = classify_priority_areas(nta_metrics, hvi_raw, hvi_threshold=hvi_threshold)

if selected_borough == "All NYC":
    nta_view = priority_df.copy()
    gardens_view = garden_joined.copy()
else:
    nta_view = priority_df[priority_df["borough"] == selected_borough].copy()
    gardens_view = garden_joined[(garden_joined["borough"] == selected_borough) | (garden_joined["nta_borough"] == selected_borough)].copy()

metric_col = {
    "Garden Access Score": "garden_access_score",
    "Garden Count": "garden_count",
    "Garden Density": "garden_density_sqmi",
}[metric_view]

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown('<div class="main-title">NYC Community Gardens for Equitable Climate Resilience</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="guiding-question">How can NYC community gardens support sustainable, equitable, and climate-resilient cities?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtle">
    Civic-tech dashboard connecting GreenThumb community gardens to SDG 11, SDG 13, SDG 15, and supporting SDG 2 and SDG 3.
    Designed for youth presenters, classroom discussion, and exploratory UN/SDG education use.
    </div>
    """,
    unsafe_allow_html=True,
)

mock_flags = [
    gardens_status.get("is_mock") == "True",
    nta_status.get("is_mock") == "True",
    pop_status.get("is_mock") == "True",
    priority_status.get("is_mock_hvi") == "True",
    clean_status.get("is_mock") == "True",
]
if any(mock_flags):
    st.markdown(
        """
        <div class="warning-callout">
        <b>Data mode notice:</b> One or more live datasets could not be loaded or matched, so the dashboard is using clearly marked mock fallback data for that layer. Replace mock population/HVI files before formal policy use.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Data loading status"):
    st.write(
        pd.DataFrame(
            [
                {"Layer": "GreenThumb gardens", **gardens_status},
                {"Layer": "Garden point cleaning", **clean_status},
                {"Layer": "2020 NTA boundaries", **nta_status},
                {"Layer": "Population", **pop_status},
                {"Layer": "HVI", **hvi_status},
                {"Layer": "Priority classification", **priority_status},
                {"Layer": "Metric computation", **metric_status},
            ]
        )
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
# Tab 1
# -----------------------------------------------------------------------------

with tab1:
    st.subheader("NYC Community Gardens Overview")
    total_gardens = len(gardens_view)
    borough_count = gardens_view["borough"].replace("Unknown", np.nan).dropna().nunique()
    active_count = gardens_view["status"].astype(str).str.contains("active", case=False, na=False).sum()
    assigned_count = gardens_view["nta_code"].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        make_kpi_card("Gardens", format_int(total_gardens), "Filtered community garden points")
    with c2:
        make_kpi_card("Boroughs", format_int(borough_count), "Represented in current filter")
    with c3:
        make_kpi_card("Active / likely active", format_int(active_count), "Based on status text when available")
    with c4:
        make_kpi_card("Spatially assigned", f"{format_int(assigned_count)}", "Gardens matched to NTAs")

    st.markdown(
        """
        <div class="callout">
        <b>Interpretation:</b> Community gardens are distributed unevenly across NYC, creating an opportunity to examine access and equity. The map below treats gardens as neighborhood-scale resilience assets rather than isolated beautification projects.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.pydeck_chart(make_garden_pydeck(gardens_view, buffer_miles=buffer_distance), use_container_width=True)

    borough_summary = (
        gardens_view.groupby("borough", dropna=False)
        .size()
        .reset_index(name="garden_count")
        .sort_values("garden_count", ascending=False)
    )
    fig_bar = px.bar(
        borough_summary,
        x="borough",
        y="garden_count",
        title="Garden count by borough",
        labels={"borough": "Borough", "garden_count": "Garden count"},
    )
    fig_bar.update_layout(height=360, paper_bgcolor=PALETTE["cream"], plot_bgcolor=PALETTE["cream"])
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2
# -----------------------------------------------------------------------------

with tab2:
    st.subheader("Access & Equity Map")
    st.markdown(
        """
        <div class="warning-callout">
        <b>Caveat:</b> Population denominators and NTA boundaries affect interpretation. This map measures neighborhood-level garden presence, not actual walking time, garden capacity, opening hours, or program quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2.1, 1])
    with col_a:
        st.plotly_chart(
            make_choropleth(
                nta_view,
                metric=metric_col,
                title=f"{metric_view} by 2020 NTA",
                color_scale="YlGn",
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("#### Metric definitions")
        st.markdown(
            """
            - **Garden Count by NTA:** number of garden points within each NTA.
            - **Garden Access Score:** gardens per 10,000 residents.
            - **Garden Density:** gardens per square mile.
            """
        )
        st.markdown("#### Bottom 10 NTAs by access score")
        bottom10 = (
            nta_view.sort_values(["garden_access_score", "garden_count"], ascending=[True, True])
            [["nta_name", "borough", "population", "garden_count", "garden_access_score"]]
            .head(10)
            .copy()
        )
        bottom10["garden_access_score"] = bottom10["garden_access_score"].round(2)
        st.dataframe(bottom10, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 3
# -----------------------------------------------------------------------------

with tab3:
    st.subheader("Climate Resilience Layer")
    st.markdown(
        """
        <div class="warning-callout">
        <b>Exploratory Analysis/Priority Mapping Only:</b> This matrix explores structural correlations, not direct causal health outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="callout">
        <b>Insight:</b> Neighborhoods with high heat vulnerability and low community garden access represent priority areas for youth-led sustainability education.
        </div>
        """,
        unsafe_allow_html=True,
    )

    map_df = nta_view.copy()
    map_df["priority_numeric"] = map_df["priority_class"].map({"Lower priority": 1, "Medium priority": 2, "High priority": 3}).fillna(1)

    col1, col2 = st.columns([2.1, 1])
    with col1:
        st.plotly_chart(
            make_choropleth(
                map_df,
                metric="hvi_rank",
                title="Heat Vulnerability Index with community garden overlay",
                color_scale="OrRd",
                overlay_points=gardens_view,
            ),
            use_container_width=True,
        )
    with col2:
        st.markdown("#### Priority classification")
        counts = map_df["priority_class"].value_counts().reindex(["High priority", "Medium priority", "Lower priority"]).fillna(0).astype(int)
        for label, count in counts.items():
            color = PRIORITY_COLORS[label]
            st.markdown(
                f"<div class='action-card' style='min-height:80px;border-left:8px solid {color};'><b>{label}</b><br/><span class='kpi-value'>{count}</span><br/><span class='subtle'>NTAs</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("#### Rule")
        st.markdown(
            f"- **High:** HVI ≥ {hvi_threshold} and access below city median  \n"
            "- **Medium:** HVI ≥ 3 and access below city median  \n"
            "- **Lower:** all other areas"
        )

    st.markdown("#### Priority neighborhoods")
    priority_table = priority_download_df(map_df)
    high_medium = priority_table[priority_table["priority_class"].isin(["High priority", "Medium priority"])]
    st.dataframe(high_medium.head(20), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 4
# -----------------------------------------------------------------------------

with tab4:
    st.subheader("SDG Lens")
    st.markdown(
        """
        Community gardens connect global goals to visible, local places. For a youth presenter, the key move is to show that SDGs are not abstract: they can be mapped, visited, measured, and improved.
        """
    )
    sdg_items = [
        {
            "sdg": "SDG 11",
            "title": "Sustainable Cities & Communities",
            "badge": "Inclusive public spaces",
            "text": "Gardens can expand neighborhood access to welcoming public space, especially where parks and tree canopy are unevenly distributed.",
        },
        {
            "sdg": "SDG 13",
            "title": "Climate Action",
            "badge": "Heat resilience awareness",
            "text": "Gardens can become learning sites for urban heat, shade, soil, stormwater, and adaptation strategies.",
        },
        {
            "sdg": "SDG 15",
            "title": "Life on Land",
            "badge": "Urban biodiversity",
            "text": "Gardens can support pollinators, habitat patches, composting, soil stewardship, and everyday biodiversity education.",
        },
        {
            "sdg": "SDG 2",
            "title": "Zero Hunger",
            "badge": "Food security education",
            "text": "Gardens do not replace food systems, but they can teach nutrition, local growing, food justice, and community care.",
        },
        {
            "sdg": "SDG 3",
            "title": "Good Health & Well-being",
            "badge": "Outdoor wellness",
            "text": "Gardens can support social connection, outdoor learning, intergenerational activity, and mental wellness.",
        },
    ]

    rows = [sdg_items[:3], sdg_items[3:]]
    for row_items in rows:
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                st.markdown(
                    f"""
                    <div class="sdg-card">
                        <span class="badge">{item['sdg']}</span>
                        <h4>{item['title']}</h4>
                        <b style="color:{PALETTE['terracotta']};">{item['badge']}</b>
                        <p>{item['text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# Tab 5
# -----------------------------------------------------------------------------

with tab5:
    st.subheader("Youth Action & Recommendations")
    selected_context = selected_borough if selected_borough != "All NYC" else "NYC"
    st.markdown(
        f"""
        These recommendations translate the map into an action agenda for **{selected_context}**. The goal is not to claim that gardens solve heat vulnerability alone; the goal is to identify where students can learn, partner, document, and advocate.
        """
    )

    priority_candidates = priority_df.copy()
    if selected_borough != "All NYC":
        priority_candidates = priority_candidates[priority_candidates["borough"] == selected_borough]
    priority_candidates = priority_candidates.sort_values(
        ["priority_sort", "hvi_rank", "garden_access_score"], ascending=[True, False, True]
    )
    priority_names = priority_candidates[priority_candidates["priority_class"].isin(["High priority", "Medium priority"])]
    if len(priority_names) == 0:
        priority_names = priority_candidates.head(5)

    top_names = priority_names["nta_name"].head(5).tolist()
    top_names_text = ", ".join(top_names) if top_names else "priority neighborhoods identified by the dashboard"

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"""
            <div class="action-card">
                <h4>1. What students can learn</h4>
                <ul>
                    <li>How local green space links to climate resilience.</li>
                    <li>Why access should be measured per resident, not only by total garden count.</li>
                    <li>How heat vulnerability and environmental justice can be mapped carefully.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="action-card">
                <h4>2. Priority neighborhoods to adopt</h4>
                <p><b>{top_names_text}</b></p>
                <p>Start with a field observation checklist: shade, seating, compost, pollinator plants, water access, educational signage, and nearby schools.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            f"""
            <div class="action-card">
                <h4>3. SDG action ideas</h4>
                <ul>
                    <li>Create a student-made SDG garden story map.</li>
                    <li>Design heat-awareness posters for garden visitors.</li>
                    <li>Run a pollinator count or soil-health mini-study.</li>
                    <li>Collect oral histories from gardeners and neighbors.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            f"""
            <div class="action-card">
                <h4>4. Community partnership ideas</h4>
                <ul>
                    <li>Partner with GreenThumb garden groups.</li>
                    <li>Invite schools to adopt nearby gardens as outdoor learning labs.</li>
                    <li>Connect youth volunteers with borough-level sustainability offices.</li>
                    <li>Share findings with community boards and local libraries.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Download priority neighborhoods")
    download_df = priority_download_df(priority_candidates)
    st.download_button(
        label="Download priority neighborhoods CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="nyc_community_garden_priority_neighborhoods.csv",
        mime="text/csv",
    )

    st.markdown("#### 60-second presenter script")
    script = f"""
Today, I am asking one question: How can NYC community gardens support sustainable, equitable, and climate-resilient cities? This dashboard treats community gardens as small but meaningful urban resilience infrastructure. First, we map where gardens are located across NYC. Then we compare access by neighborhood, using gardens per 10,000 residents and gardens per square mile. Next, we add heat vulnerability. The most important areas are not simply the places with the fewest gardens. They are places where heat vulnerability is high and garden access is below the city median. For {selected_context}, this helps identify neighborhoods where youth-led sustainability education could matter most. The SDG lens connects this local map to SDG 11 for inclusive public space, SDG 13 for climate adaptation, SDG 15 for biodiversity, and supporting SDG 2 and SDG 3 through food and wellness education. This is not a causal health study. It is a priority map that helps students ask better questions, build partnerships, and turn data into community action.
""".strip()
    st.text_area("Presenter script", script, height=230)

# -----------------------------------------------------------------------------
# Footer caveats
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"""
    <div class="subtle">
    <b>Data caveats:</b> This is an exploratory dashboard. It does not estimate direct health outcomes, walking-network access, garden capacity, opening hours, stewardship quality, or causal climate effects. Area and density are approximated without projected GIS libraries to improve Streamlit Cloud deployability. For formal analysis, replace mock population data with a verified NTA population CSV and validate HVI-to-NTA joins. {WGS84_NOTE}
    </div>
    """,
    unsafe_allow_html=True,
)
