"""
NYC Community Gardens for Equitable Climate Resilience

Deployment-stable Streamlit dashboard built with pandas, shapely, plotly, and pydeck.
This version intentionally avoids geopandas/fiona/GDAL because those packages often fail
on Streamlit Community Cloud builds.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from shapely import wkt
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform
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
# Constants
# -----------------------------------------------------------------------------

GREENTHUMB_ID = "p78i-pat6"
NTA_ID = "9nt8-h7nd"
HVI_ID = "4mhf-duep"

SODA_JSON = "https://data.cityofnewyork.us/resource/{dataset_id}.json?$limit=50000"
SODA_GEOJSON = "https://data.cityofnewyork.us/api/views/{dataset_id}/rows.geojson?accessType=DOWNLOAD"
SODA_GEOSPATIAL_EXPORT = (
    "https://data.cityofnewyork.us/api/geospatial/{dataset_id}?method=export&format=GeoJSON"
)
CDTA_HVI_ARCGIS = (
    "https://services1.arcgis.com/8cuieNI8NbqQZQVJ/ArcGIS/rest/services/"
    "HVI_by_CDTA_CRAD_2024/FeatureServer/0/query?where=1%3D1&outFields=*&f=json"
)

NYC_CENTER = {"lat": 40.7128, "lon": -74.0060}
EARTH_RADIUS_FT = 20_925_524.9
AREA_SQFT_PER_SQMI = 27_878_400

# Higher-contrast, more legible civic-tech palette.
PALETTE = {
    "forest": "#0E3327",
    "forest_2": "#164B3A",
    "sage": "#DCE8D5",
    "sage_dark": "#6F8F76",
    "cream": "#FAF8F1",
    "white": "#FFFFFF",
    "terracotta": "#B85C38",
    "terracotta_light": "#FFF0E8",
    "ink": "#18251F",
    "muted": "#56645E",
    "line": "#D9DED3",
}

BOROUGH_ORDER = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
BOROUGH_ABBR = {
    "Bronx": "BX",
    "Brooklyn": "BK",
    "Manhattan": "MN",
    "Queens": "QN",
    "Staten Island": "SI",
}

DATASET_FACTORS = [
    {
        "Dataset": "GreenThumb Garden Info",
        "Source ID": GREENTHUMB_ID,
        "What goes into it": "Garden identity and location records maintained by NYC Parks GreenThumb.",
        "Fields used in dashboard": "garden/site name, borough, status when available, address when available, geometry or latitude/longitude converted to a point.",
        "Dashboard role": "Maps garden locations; counts gardens by borough and by NTA.",
    },
    {
        "Dataset": "2020 Neighborhood Tabulation Areas",
        "Source ID": NTA_ID,
        "What goes into it": "2020 NTA statistical geography polygons created by NYC Planning from 2020 census tract groupings.",
        "Fields used in dashboard": "NTA code, NTA name, borough, polygon geometry.",
        "Dashboard role": "Defines neighborhood boundaries for spatial join, access score, density, and choropleth maps.",
    },
    {
        "Dataset": "NTA Population",
        "Source ID": "local CSV: data/nta_population.csv",
        "What goes into it": "2020 Census total population by residential NTA, stored locally for stable deployment.",
        "Fields used in dashboard": "nta_code, nta_name, population.",
        "Dashboard role": "Creates Garden Access Score = gardens per 10,000 residents.",
    },
    {
        "Dataset": "Heat Vulnerability Index",
        "Source ID": HVI_ID,
        "What goes into it": "NYC Health HVI summarizes heat risk using social and environmental factors.",
        "Fields used in dashboard": "HVI rank 1 to 5; CDTA proxy is used when direct NTA matching is unavailable.",
        "Dashboard role": "Creates climate vulnerability layer and priority classification.",
    },
]

HVI_FACTOR_ROWS = [
    {"HVI factor": "Median household income", "Interpretation": "Lower income can reduce ability to afford and use cooling resources."},
    {"HVI factor": "Percent vegetative cover", "Interpretation": "Tree, grass, and shrub cover can reduce neighborhood heat exposure."},
    {"HVI factor": "Percent Non-Hispanic Black population", "Interpretation": "Included by NYC Health because heat impacts reflect structural inequities and historic disinvestment."},
    {"HVI factor": "Average surface temperature", "Interpretation": "Hotter land-surface temperatures are associated with greater heat-wave mortality risk."},
    {"HVI factor": "Percent households reporting air-conditioning access", "Interpretation": "AC access is protective during extreme heat events."},
]

METRIC_EXPLANATIONS = {
    "garden_count": {
        "label": "Garden Count",
        "plain": "How many GreenThumb garden point records fall inside each NTA boundary.",
        "formula": "count(garden points within NTA)",
        "use": "Best for seeing where gardens are physically present or absent.",
        "watchout": "It is not population-adjusted; a large neighborhood and a small neighborhood can look equal if both have the same count.",
    },
    "garden_access_score": {
        "label": "Garden Access Score",
        "plain": "How many community gardens exist for every 10,000 residents in an NTA.",
        "formula": "garden_count / population × 10,000",
        "use": "Best for equity analysis because it compares garden availability against neighborhood population size.",
        "watchout": "It does not measure walking distance, garden capacity, opening hours, or program quality.",
    },
    "garden_density_per_sq_mile": {
        "label": "Garden Density",
        "plain": "How concentrated gardens are within the physical area of an NTA.",
        "formula": "garden_count / NTA land area in square miles",
        "use": "Best for understanding spatial concentration of green infrastructure.",
        "watchout": "It does not account for how many people live in the area or who has access to the garden.",
    },
    "hvi_rank": {
        "label": "Heat Vulnerability Index",
        "plain": "A relative heat-risk ranking used to identify communities more vulnerable to extreme heat.",
        "formula": "NYC Health index rank, commonly interpreted from 1 = lower risk to 5 = higher risk",
        "use": "Best for climate-resilience prioritization when paired with access and equity measures.",
        "watchout": "It is an index for exploratory mapping, not a direct causal health-outcome measure.",
    },
    "priority_class": {
        "label": "Priority Classification",
        "plain": "A simple action-oriented grouping that combines heat vulnerability with garden access.",
        "formula": "High = HVI ≥ selected threshold and access below city median; Medium = HVI ≥ 3 and access below city median; Lower = all others",
        "use": "Best for selecting neighborhoods for youth-led SDG education and partnership outreach.",
        "watchout": "It is a prioritization lens, not a final funding or policy decision rule.",
    },
}

METRIC_DICTIONARY_ROWS = [
    {
        "Metric": item["label"],
        "What it explains": item["plain"],
        "Calculation / rule": item["formula"],
        "Best used for": item["use"],
    }
    for item in METRIC_EXPLANATIONS.values()
]

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {PALETTE['cream']};
            color: {PALETTE['ink']};
        }}
        .block-container {{
            padding-top: 2rem;
            max-width: 1340px;
        }}
        [data-testid="stSidebar"] {{
            background: {PALETTE['forest']};
            border-right: 1px solid rgba(255,255,255,0.12);
        }}
        [data-testid="stSidebar"] * {{
            color: #FDFBF4 !important;
        }}
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stSlider div,
        [data-testid="stSidebar"] .stRadio div {{
            color: #FDFBF4 !important;
        }}
        .main-title {{
            font-size: 2.55rem;
            line-height: 1.05;
            font-weight: 850;
            color: {PALETTE['forest']};
            margin-bottom: 0.15rem;
            letter-spacing: -0.035em;
        }}
        .guiding-question {{
            font-size: 1.22rem;
            font-weight: 700;
            color: {PALETTE['ink']};
            background: {PALETTE['white']};
            border-left: 7px solid {PALETTE['terracotta']};
            border: 1px solid {PALETTE['line']};
            padding: 1rem 1.1rem;
            border-radius: 0.8rem;
            margin: 0.75rem 0 1.05rem 0;
            box-shadow: 0 6px 22px rgba(14,51,39,0.06);
        }}
        .eyebrow {{
            text-transform: uppercase;
            letter-spacing: .11em;
            font-size: .78rem;
            color: {PALETTE['sage_dark']};
            font-weight: 850;
            margin-bottom: .25rem;
        }}
        .subtle {{
            color: {PALETTE['muted']};
            font-size: 0.96rem;
        }}
        .kpi-card {{
            background: {PALETTE['white']};
            border: 1px solid {PALETTE['line']};
            border-radius: 1rem;
            padding: 1rem 1.05rem;
            box-shadow: 0 5px 16px rgba(14, 51, 39, 0.08);
            min-height: 118px;
        }}
        .kpi-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: {PALETTE['muted']};
            font-weight: 850;
        }}
        .kpi-value {{
            font-size: 2.05rem;
            color: {PALETTE['forest']};
            font-weight: 900;
            margin-top: 0.25rem;
        }}
        .callout {{
            background: {PALETTE['sage']};
            border-left: 6px solid {PALETTE['forest']};
            border-radius: 0.85rem;
            padding: 0.95rem 1.05rem;
            margin: 0.75rem 0;
            color: {PALETTE['ink']};
            border-top: 1px solid rgba(14,51,39,0.10);
            border-right: 1px solid rgba(14,51,39,0.10);
            border-bottom: 1px solid rgba(14,51,39,0.10);
        }}
        .warning-callout {{
            background: {PALETTE['terracotta_light']};
            border-left: 6px solid {PALETTE['terracotta']};
            border-radius: 0.85rem;
            padding: 0.95rem 1.05rem;
            margin: 0.75rem 0;
            color: {PALETTE['ink']};
            border-top: 1px solid rgba(184,92,56,0.22);
            border-right: 1px solid rgba(184,92,56,0.22);
            border-bottom: 1px solid rgba(184,92,56,0.22);
        }}
        .action-card, .sdg-card {{
            background: {PALETTE['white']};
            border: 1px solid {PALETTE['line']};
            border-radius: 1rem;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 16px rgba(14, 51, 39, 0.07);
            min-height: 168px;
        }}
        .badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.34rem 0.7rem;
            margin: 0.12rem 0.2rem 0.12rem 0;
            background: {PALETTE['forest']};
            color: #FFFFFF;
            font-weight: 850;
            font-size: 0.82rem;
        }}
        .metric-pill {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            margin: 0.15rem 0.15rem 0.15rem 0;
            background: #EEF4EA;
            color: {PALETTE['forest']};
            border: 1px solid {PALETTE['line']};
            font-weight: 750;
            font-size: 0.82rem;
        }}
        div[data-testid="stMetric"] {{
            background: {PALETTE['white']};
            padding: 0.85rem;
            border-radius: 0.9rem;
            border: 1px solid {PALETTE['line']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: .35rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: #FFFFFF;
            border: 1px solid {PALETTE['line']};
            border-radius: .75rem .75rem 0 0;
            padding: .75rem .9rem;
            color: {PALETTE['ink']};
            font-weight: 750;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PALETTE['forest']} !important;
            color: #FFFFFF !important;
        }}
        .hero-panel {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #FFFFFF 0%, #F3F7EF 58%, #FFF2EA 100%);
            border: 1px solid {PALETTE['line']};
            border-radius: 1.35rem;
            padding: 1.35rem 1.45rem 1.15rem 1.45rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(14, 51, 39, 0.11);
        }}
        .hero-panel::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, {PALETTE['forest']}, {PALETTE['sage_dark']}, {PALETTE['terracotta']});
        }}
        .hero-meta {{
            display: flex;
            gap: .45rem;
            flex-wrap: wrap;
            margin-top: .8rem;
        }}
        .source-chip {{
            display: inline-block;
            border-radius: 999px;
            background: rgba(14, 51, 39, 0.07);
            border: 1px solid rgba(14, 51, 39, 0.13);
            color: {PALETTE['forest']};
            font-size: .78rem;
            font-weight: 820;
            letter-spacing: .015em;
            padding: .35rem .65rem;
        }}
        .metric-explainer {{
            background: #FFFFFF;
            border: 1px solid {PALETTE['line']};
            border-left: 6px solid {PALETTE['terracotta']};
            border-radius: .95rem;
            padding: .9rem 1rem;
            margin: .75rem 0 1rem 0;
            box-shadow: 0 7px 18px rgba(14, 51, 39, 0.07);
            color: {PALETTE['ink']};
        }}
        .metric-explainer h4 {{
            margin: 0 0 .35rem 0;
            color: {PALETTE['forest']};
        }}
        .metric-explainer p {{
            margin: .25rem 0;
        }}
        .sidebar-note {{
            background: rgba(255,255,255,0.94);
            border-radius: .9rem;
            padding: .78rem .82rem;
            border: 1px solid rgba(255,255,255,0.42);
            color: {PALETTE['ink']} !important;
            margin-top: .7rem;
        }}
        .sidebar-note * {{
            color: {PALETTE['ink']} !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            border: 1px solid rgba(14,51,39,.18) !important;
            border-radius: .65rem !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] input,
        [data-testid="stSidebar"] div[data-baseweb="select"] svg {{
            color: #111111 !important;
            fill: #111111 !important;
        }}
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div[role="option"],
        div[data-baseweb="popover"] * {{
            color: #111111 !important;
        }}
        hr {{
            border: none;
            border-top: 1px solid {PALETTE['line']};
            margin: 1.25rem 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def normalize_col_name(col: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(col).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col_name(c) for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    existing = set(df.columns)
    for col in candidates:
        c = normalize_col_name(col)
        if c in existing:
            return c
    return None


def normalize_text_key(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value)


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


def request_json(url: str, timeout: int = 30):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def lonlat_to_local_feet(lon: float, lat: float) -> Tuple[float, float]:
    lat0 = math.radians(NYC_CENTER["lat"])
    x = math.radians(lon - NYC_CENTER["lon"]) * EARTH_RADIUS_FT * math.cos(lat0)
    y = math.radians(lat - NYC_CENTER["lat"]) * EARTH_RADIUS_FT
    return x, y


def approximate_area_sq_mi(geom) -> float:
    try:
        projected = transform(lambda x, y, z=None: lonlat_to_local_feet(x, y), geom)
        return float(projected.area / AREA_SQFT_PER_SQMI)
    except Exception:
        return np.nan


def city_median(series: pd.Series) -> float:
    med = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    return 0.0 if pd.isna(med) else float(med)


def format_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def get_borough_options(df: pd.DataFrame) -> List[str]:
    if "borough" not in df.columns:
        return ["All boroughs"]
    found = [b for b in BOROUGH_ORDER if b in set(df["borough"].dropna().astype(str))]
    extra = sorted(set(df["borough"].dropna().astype(str)) - set(found))
    return ["All boroughs"] + found + extra


def filtered_by_borough(df: pd.DataFrame, borough: str) -> pd.DataFrame:
    if borough == "All boroughs" or "borough" not in df.columns:
        return df.copy()
    return df[df["borough"].astype(str) == borough].copy()


def friendly_metric_name(metric: str) -> str:
    names = {
        "garden_count": "Garden Count",
        "garden_access_score": "Gardens per 10,000 Residents",
        "garden_density_per_sq_mile": "Gardens per Square Mile",
        "hvi_rank": "Heat Vulnerability Index",
    }
    return names.get(metric, metric.replace("_", " ").title())


def render_selected_metric_explainer(metric: str, compact: bool = False) -> None:
    info = METRIC_EXPLANATIONS.get(metric)
    if not info:
        return
    if compact:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-note">
                <strong>{info['label']}</strong><br>
                {info['plain']}<br>
                <span style="font-size:.82rem;"><strong>Formula:</strong> {info['formula']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="metric-explainer">
            <h4>{info['label']}</h4>
            <p><strong>What it explains:</strong> {info['plain']}</p>
            <p><strong>Calculation / rule:</strong> {info['formula']}</p>
            <p><strong>Best used for:</strong> {info['use']}</p>
            <p><strong>Interpretation caution:</strong> {info['watchout']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def df_to_geojson(df: pd.DataFrame) -> Dict:
    features = []
    for _, row in df.iterrows():
        geom = row.get("geometry")
        if geom is None or getattr(geom, "is_empty", True):
            continue
        props = {k: v for k, v in row.items() if k != "geometry"}
        clean_props = {}
        for k, v in props.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_props[k] = float(v) if not pd.isna(v) else None
            elif pd.isna(v):
                clean_props[k] = None
            else:
                clean_props[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        features.append({"type": "Feature", "properties": clean_props, "geometry": mapping(geom)})
    return {"type": "FeatureCollection", "features": features}

# -----------------------------------------------------------------------------
# Mock fallbacks
# -----------------------------------------------------------------------------


def create_mock_nta_data() -> pd.DataFrame:
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
    return pd.DataFrame(records)


def create_mock_garden_data() -> pd.DataFrame:
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
            point = Point(lon + random.uniform(-0.045, 0.045), lat + random.uniform(-0.035, 0.035))
            rows.append(
                {
                    "garden_id": f"MOCK-G{garden_id:04d}",
                    "garden_name": f"Mock {borough} Community Garden {i + 1}",
                    "borough": borough,
                    "status": "Active" if i % 7 != 0 else "Other / Unknown",
                    "address": "Mock address for offline fallback",
                    "geometry": point,
                    "lon": point.x,
                    "lat": point.y,
                    "mock_garden": True,
                }
            )
            garden_id += 1
    return pd.DataFrame(rows)


def create_mock_population_data(nta_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    borough_base = {
        "Bronx": 41000,
        "Brooklyn": 52000,
        "Manhattan": 61000,
        "Queens": 48000,
        "Staten Island": 33000,
    }
    df = nta_df[["nta_code", "nta_name", "borough"]].copy()
    df["population"] = [
        int(borough_base.get(b, 45000) + rng.integers(-12000, 15000)) for b in df["borough"]
    ]
    df["population_source"] = "mock_placeholder"
    return df


def create_mock_hvi_data(nta_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = {"Bronx": 4, "Brooklyn": 3, "Manhattan": 2, "Queens": 3, "Staten Island": 2}
    df = nta_df[["nta_code", "nta_name", "borough"]].copy()
    df["hvi_rank"] = [int(np.clip(base.get(b, 3) + rng.choice([-1, 0, 0, 1]), 1, 5)) for b in df["borough"]]
    df["hvi_source"] = "mock_placeholder"
    return df

# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_greenthumb_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    status = {
        "dataset": "GreenThumb Garden Info",
        "dataset_id": GREENTHUMB_ID,
        "source": "NYC Open Data GeoJSON/API",
        "mock": "False",
        "message": "Loaded official garden records from NYC Open Data.",
    }
    try:
        geo = request_json(SODA_GEOJSON.format(dataset_id=GREENTHUMB_ID))
        rows = []
        for i, feat in enumerate(geo.get("features", [])):
            props = feat.get("properties", {}) or {}
            geom_raw = feat.get("geometry")
            geom = shape(geom_raw) if geom_raw else None
            props["geometry"] = geom
            props["source_row"] = i
            rows.append(props)
        df = normalize_columns(pd.DataFrame(rows))
        if len(df) == 0:
            raise ValueError("GreenThumb GeoJSON returned no features.")
        return df, status
    except Exception as geo_error:
        try:
            records = request_json(SODA_JSON.format(dataset_id=GREENTHUMB_ID))
            df = normalize_columns(pd.DataFrame(records))
            status["source"] = "NYC Open Data JSON"
            status["message"] = f"Loaded garden records from JSON after GeoJSON issue: {geo_error}"
            return df, status
        except Exception as exc:
            status.update({"source": "Mock fallback", "mock": "True", "message": f"Live garden data unavailable. Using mock sample. Error: {exc}"})
            return create_mock_garden_data(), status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_nta_boundaries() -> Tuple[pd.DataFrame, Dict[str, str]]:
    status = {
        "dataset": "2020 Neighborhood Tabulation Areas",
        "dataset_id": NTA_ID,
        "source": "NYC Open Data geospatial export",
        "mock": "False",
        "message": "Loaded official 2020 NTA boundaries.",
    }
    last_error = None
    for url in [SODA_GEOSPATIAL_EXPORT.format(dataset_id=NTA_ID), SODA_GEOJSON.format(dataset_id=NTA_ID)]:
        try:
            geo = request_json(url)
            rows = []
            for i, feat in enumerate(geo.get("features", [])):
                props = feat.get("properties", {}) or {}
                geom_raw = feat.get("geometry")
                if not geom_raw:
                    continue
                props["geometry"] = shape(geom_raw)
                props["source_row"] = i
                rows.append(props)
            raw = normalize_columns(pd.DataFrame(rows))
            if len(raw) == 0:
                raise ValueError("NTA endpoint returned no features.")

            code_col = first_existing(raw, ["nta2020", "nta2020_code", "ntacode", "nta_code", "geoid", "nta", "ntacode2020"])
            name_col = first_existing(raw, ["ntaname", "nta_name", "nta2020_name", "name", "neighborhood", "ntaname2020"])
            boro_col = first_existing(raw, ["boroname", "boro_name", "borough", "boro", "borocode", "boro_code"])
            out = pd.DataFrame()
            out["nta_code"] = raw[code_col].astype(str) if code_col else [f"NTA_{i:04d}" for i in range(len(raw))]
            out["nta_name"] = raw[name_col].astype(str) if name_col else out["nta_code"]
            out["borough"] = raw[boro_col].apply(standardize_borough) if boro_col else "Unknown"
            out["geometry"] = raw["geometry"]
            out["mock_nta"] = False
            return out, status
        except Exception as exc:
            last_error = exc
    status.update({"source": "Mock fallback", "mock": "True", "message": f"Live NTA boundaries unavailable. Using mock boundaries. Error: {last_error}"})
    return create_mock_nta_data(), status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_population_data(nta_reference: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    status = {
        "dataset": "NTA population",
        "dataset_id": "data/nta_population.csv",
        "source": "Local CSV",
        "mock": "False",
        "message": "Loaded local 2020 NTA population CSV.",
    }
    for path in [Path("data/nta_population.csv"), Path("nta_population.csv")]:
        try:
            df = normalize_columns(pd.read_csv(path))
            pop_col = first_existing(df, ["population", "pop", "total_population", "pop_2020"])
            code_col = first_existing(df, ["nta_code", "ntacode", "nta2020", "geoid"])
            name_col = first_existing(df, ["nta_name", "ntaname", "nta2020_name", "name"])
            if pop_col is None or (code_col is None and name_col is None):
                raise ValueError("Population CSV requires population plus nta_code or nta_name.")
            out = pd.DataFrame()
            if code_col:
                out["nta_code"] = df[code_col].astype(str)
            if name_col:
                out["nta_name"] = df[name_col].astype(str)
            out["population"] = pd.to_numeric(df[pop_col], errors="coerce")
            out = out.dropna(subset=["population"])
            out["population_source"] = str(path)
            status["message"] = f"Loaded local population file: {path}"
            return out, status
        except Exception:
            continue
    if nta_reference is not None and len(nta_reference) > 0:
        mock = create_mock_population_data(nta_reference)
    else:
        mock = pd.DataFrame(columns=["nta_code", "nta_name", "population", "population_source"])
    status.update({"source": "Mock fallback", "mock": "True", "message": "Population file not found or unreadable. Using mock denominators."})
    return mock, status


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_hvi_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load HVI. Preferred: local NTA HVI. Secondary: official NYC Health CDTA proxy. Tertiary: Open Data HVI table."""
    status = {
        "dataset": "Heat Vulnerability Index",
        "dataset_id": HVI_ID,
        "source": "NYC Open Data / NYC Health",
        "mock": "False",
        "message": "Loaded HVI records.",
    }

    # Optional local NTA-level file, if later added.
    for path in [Path("data/hvi_nta.csv"), Path("hvi_nta.csv")]:
        try:
            if path.exists():
                df = normalize_columns(pd.read_csv(path))
                rank_col = first_existing(df, ["hvi_rank", "hvi", "heat_vulnerability_index", "rank"])
                code_col = first_existing(df, ["nta_code", "ntacode", "nta2020", "geoid"])
                name_col = first_existing(df, ["nta_name", "ntaname", "name"])
                if rank_col is None or (code_col is None and name_col is None):
                    raise ValueError("Local HVI CSV requires hvi_rank plus nta_code or nta_name.")
                out = pd.DataFrame()
                if code_col:
                    out["nta_code"] = df[code_col].astype(str)
                if name_col:
                    out["nta_name"] = df[name_col].astype(str)
                    out["nta_name_key"] = out["nta_name"].map(normalize_text_key)
                out["hvi_rank"] = pd.to_numeric(df[rank_col], errors="coerce").clip(1, 5)
                out = out.dropna(subset=["hvi_rank"])
                out["hvi_source"] = str(path)
                status.update({"source": str(path), "message": f"Loaded local NTA-level HVI file: {path}"})
                return out, status
        except Exception:
            continue

    # Official CDTA-level HVI proxy from NYC Health/ArcGIS.
    try:
        data = request_json(CDTA_HVI_ARCGIS)
        features = data.get("features", [])
        rows = [f.get("attributes", {}) for f in features]
        raw = normalize_columns(pd.DataFrame(rows))
        if len(raw) == 0:
            raise ValueError("CDTA HVI endpoint returned no rows.")
        rank_col = first_existing(raw, ["hvi", "hvi_rank", "heat_vulnerability_index", "rank"])
        comm_col = first_existing(raw, ["commdist", "cdta", "cdta2020", "community_district", "community_district_tabulation_area"])
        boro_col = first_existing(raw, ["borough", "boro", "boroname"])
        if rank_col is None:
            raise ValueError("Could not identify HVI rank field in CDTA endpoint.")
        out = pd.DataFrame()
        out["hvi_rank"] = pd.to_numeric(raw[rank_col], errors="coerce").clip(1, 5)
        if comm_col:
            out["cdta_code"] = raw[comm_col].astype(str).str.upper().str.extract(r"([A-Z]{2}\s*\d{1,2})", expand=False)
            # If the field is numeric like 301 and borough is available, create BK01-style code.
            if out["cdta_code"].isna().all() and boro_col:
                boro = raw[boro_col].apply(standardize_borough).map(BOROUGH_ABBR)
                nums = raw[comm_col].astype(str).str.extract(r"(\d{1,2})$", expand=False).str.zfill(2)
                out["cdta_code"] = boro + nums
            out["cdta_code"] = out["cdta_code"].str.replace(" ", "", regex=False)
        else:
            raise ValueError("Could not identify CDTA/community district field.")
        out = out.dropna(subset=["hvi_rank", "cdta_code"])
        if len(out) == 0:
            raise ValueError("CDTA HVI records could not be standardized.")
        out["hvi_source"] = "NYC Health CDTA HVI proxy"
        status.update({"source": "NYC Health CDTA HVI proxy", "message": "Loaded official CDTA-level HVI and will assign to NTAs by parent CDTA code."})
        return out, status
    except Exception as cdta_error:
        # Keep official Open Data HVI for transparency, even if it cannot join to NTA directly.
        try:
            records = request_json(SODA_JSON.format(dataset_id=HVI_ID))
            raw = normalize_columns(pd.DataFrame(records))
            if len(raw) == 0:
                raise ValueError("Open Data HVI endpoint returned no rows.")
            rank_col = first_existing(raw, ["hvi_rank", "hvi", "heat_vulnerability_index", "rank", "score"])
            out = raw.copy()
            if rank_col:
                out["hvi_rank"] = pd.to_numeric(out[rank_col], errors="coerce").clip(1, 5)
            out["hvi_source"] = f"NYC Open Data {HVI_ID}; direct NTA join may be unavailable"
            status.update({"source": f"NYC Open Data {HVI_ID}", "message": f"Loaded HVI table after CDTA proxy failed. Direct NTA join may be unavailable. CDTA error: {cdta_error}"})
            return out, status
        except Exception as exc:
            status.update({"source": "Mock fallback pending NTA join", "mock": "True", "message": f"HVI unavailable. Mock fallback will be used. Error: {exc}"})
            return pd.DataFrame(), status

# -----------------------------------------------------------------------------
# Cleaning and metric engineering
# -----------------------------------------------------------------------------


def standardize_garden_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    name_col = first_existing(df, ["gardenname", "garden_name", "name", "site_name", "garden", "park_name", "propertyname"])
    boro_col = first_existing(df, ["borough", "boro", "boroname", "boro_name", "county"])
    status_col = first_existing(df, ["status", "garden_status", "active", "operational_status"])
    address_col = first_existing(df, ["address", "location", "street_address", "full_address"])
    id_col = first_existing(df, ["garden_id", "gispropnum", "property_id", "objectid", "source_row"])

    out = pd.DataFrame()
    out["garden_id"] = df[id_col].astype(str) if id_col else [f"G{i:05d}" for i in range(len(df))]
    out["garden_name"] = df[name_col].astype(str) if name_col else [f"Community Garden {i + 1}" for i in range(len(df))]
    out["borough"] = df[boro_col].apply(standardize_borough) if boro_col else None
    out["status"] = df[status_col].astype(str).fillna("Unknown") if status_col else "Unknown"
    out["address"] = df[address_col].astype(str).fillna("") if address_col else ""
    out["geometry"] = df["geometry"]
    out["lon"] = out["geometry"].apply(lambda g: g.x if isinstance(g, Point) else np.nan)
    out["lat"] = out["geometry"].apply(lambda g: g.y if isinstance(g, Point) else np.nan)
    out["mock_garden"] = df["mock_garden"] if "mock_garden" in df.columns else False
    return out.dropna(subset=["lon", "lat"]).copy()


def clean_garden_points(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    status = {"mock": "False", "message": "Garden locations cleaned from geometry or coordinates."}
    if raw_data is None or len(raw_data) == 0:
        status.update({"mock": "True", "message": "No garden data supplied. Using mock garden points."})
        return create_mock_garden_data(), status

    df = normalize_columns(raw_data)
    geometries = []

    if "geometry" in df.columns and df["geometry"].notna().any():
        for geom in df["geometry"]:
            if geom is None or pd.isna(geom):
                geometries.append(None)
            elif isinstance(geom, Point):
                geometries.append(geom)
            else:
                try:
                    geometries.append(geom.representative_point())
                except Exception:
                    geometries.append(None)
        df["geometry"] = geometries
        cleaned = standardize_garden_columns(df.dropna(subset=["geometry"]))
        if len(cleaned) > 0:
            return cleaned, status

    lat_col = first_existing(df, ["latitude", "lat", "y", "garden_latitude"])
    lon_col = first_existing(df, ["longitude", "lon", "lng", "long", "x", "garden_longitude"])
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        valid = lat.between(40.45, 40.95) & lon.between(-74.35, -73.65)
        if valid.any():
            df = df.loc[valid].copy()
            df["geometry"] = [Point(xy) for xy in zip(lon.loc[valid], lat.loc[valid])]
            return standardize_garden_columns(df), status

    geom_col = first_existing(df, ["multipolygon", "polygon", "the_geom", "geom", "geometry"])
    if geom_col:
        parsed = []
        for value in df[geom_col]:
            geom = None
            try:
                if isinstance(value, str) and any(token in value.upper() for token in ["POINT", "POLYGON", "MULTIPOLYGON"]):
                    geom = wkt.loads(value)
                elif isinstance(value, dict):
                    geom = shape(value)
            except Exception:
                geom = None
            if geom is not None and not isinstance(geom, Point):
                try:
                    geom = geom.representative_point()
                except Exception:
                    geom = None
            parsed.append(geom)
        df["geometry"] = parsed
        cleaned = standardize_garden_columns(df.dropna(subset=["geometry"]))
        if len(cleaned) > 0:
            return cleaned, status

    status.update({"mock": "True", "message": "Latitude/longitude and geometry were missing or unreadable. Using mock garden points."})
    return create_mock_garden_data(), status


def compute_nta_metrics(nta_boundaries: pd.DataFrame, garden_points: pd.DataFrame, population_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    status = {"message": "Computed NTA access metrics.", "population_mock": "False"}
    nta = nta_boundaries.copy()
    gardens = garden_points.copy()

    nta["area_sq_mi"] = nta["geometry"].apply(approximate_area_sq_mi)

    prepared_areas = []
    for _, row in nta.iterrows():
        geom = row["geometry"]
        if geom is not None:
            prepared_areas.append((row["nta_code"], row["nta_name"], row["borough"], geom, prep(geom)))

    joined_rows = []
    for _, garden in gardens.iterrows():
        found = {"nta_code": np.nan, "nta_name": np.nan, "nta_borough": np.nan}
        pt = garden["geometry"]
        for code, name, borough, geom, prepared in prepared_areas:
            try:
                if prepared.contains(pt) or geom.intersects(pt):
                    found = {"nta_code": code, "nta_name": name, "nta_borough": borough}
                    break
            except Exception:
                continue
        joined_rows.append(found)
    joined = pd.concat([gardens.reset_index(drop=True), pd.DataFrame(joined_rows)], axis=1)
    if "nta_borough" in joined.columns:
        joined["borough"] = joined["borough"].combine_first(joined["nta_borough"])

    counts = joined.dropna(subset=["nta_code"]).groupby("nta_code").size().rename("garden_count")
    nta = nta.merge(counts, left_on="nta_code", right_index=True, how="left")
    nta["garden_count"] = nta["garden_count"].fillna(0).astype(int)

    pop = normalize_columns(population_df) if population_df is not None else pd.DataFrame()
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
        nta = nta.drop(columns=["population"], errors="ignore").merge(mock_pop[["nta_code", "population"]], on="nta_code", how="left")
        status.update({"population_mock": "True", "message": "Population join was incomplete. Mock placeholders were used for population denominators."})

    nta["population"] = pd.to_numeric(nta["population"], errors="coerce").fillna(0)
    nta["garden_access_score"] = np.where(nta["population"] > 0, nta["garden_count"] / nta["population"] * 10_000, 0)
    nta["garden_density_per_sq_mile"] = np.where(nta["area_sq_mi"] > 0, nta["garden_count"] / nta["area_sq_mi"], 0)
    nta["access_below_city_median"] = nta["garden_access_score"] < city_median(nta["garden_access_score"])

    return nta, joined, status


def classify_priority_areas(nta_metrics: pd.DataFrame, hvi_df: pd.DataFrame, hvi_threshold: int = 4) -> Tuple[pd.DataFrame, Dict[str, str]]:
    status = {"hvi_mock": "False", "message": "Joined HVI data to NTA metrics.", "join_method": "none"}
    gdf = nta_metrics.copy()
    hvi = normalize_columns(hvi_df) if hvi_df is not None else pd.DataFrame()

    joined = False
    if len(hvi) > 0:
        rank_col = first_existing(hvi, ["hvi_rank", "hvi", "heat_vulnerability_index", "rank", "score"])
        if rank_col and rank_col != "hvi_rank":
            hvi["hvi_rank"] = pd.to_numeric(hvi[rank_col], errors="coerce").clip(1, 5)
        if "hvi_rank" in hvi.columns:
            hvi["hvi_rank"] = pd.to_numeric(hvi["hvi_rank"], errors="coerce").clip(1, 5)

            if "nta_code" in hvi.columns:
                tmp = hvi[["nta_code", "hvi_rank", "hvi_source"]].dropna(subset=["hvi_rank"]).drop_duplicates("nta_code")
                gdf = gdf.merge(tmp, on="nta_code", how="left")
                joined = gdf["hvi_rank"].notna().mean() >= 0.25
                status["join_method"] = "nta_code"

            if (not joined) and "cdta_code" in hvi.columns:
                gdf = gdf.drop(columns=["hvi_rank", "hvi_source"], errors="ignore")
                gdf["cdta_code"] = gdf["nta_code"].astype(str).str[:4].str.upper()
                tmp = hvi[["cdta_code", "hvi_rank", "hvi_source"]].dropna(subset=["hvi_rank"]).drop_duplicates("cdta_code")
                gdf = gdf.merge(tmp, on="cdta_code", how="left")
                joined = gdf["hvi_rank"].notna().mean() >= 0.25
                status.update({"join_method": "cdta_proxy", "message": "Joined official CDTA-level HVI to NTAs by parent CDTA code for exploratory proxy mapping."})

            if (not joined) and "nta_name_key" in hvi.columns:
                gdf = gdf.drop(columns=["hvi_rank", "hvi_source"], errors="ignore")
                gdf["nta_name_key"] = gdf["nta_name"].map(normalize_text_key)
                tmp = hvi[["nta_name_key", "hvi_rank", "hvi_source"]].dropna(subset=["hvi_rank"]).drop_duplicates("nta_name_key")
                gdf = gdf.merge(tmp, on="nta_name_key", how="left")
                joined = gdf["hvi_rank"].notna().mean() >= 0.25
                status["join_method"] = "nta_name"

    if not joined:
        gdf = gdf.drop(columns=["hvi_rank", "hvi_source"], errors="ignore")
        mock = create_mock_hvi_data(gdf)
        gdf = gdf.merge(mock[["nta_code", "hvi_rank", "hvi_source"]], on="nta_code", how="left")
        status.update({"hvi_mock": "True", "join_method": "mock_by_nta", "message": "HVI could not be matched to NTA geography. Mock HVI placeholders were used."})

    median_access = city_median(gdf["garden_access_score"])
    gdf["access_below_city_median"] = gdf["garden_access_score"] < median_access
    high_cut = int(hvi_threshold)
    medium_cut = max(3, high_cut - 1)
    gdf["priority_class"] = np.select(
        [
            (gdf["hvi_rank"] >= high_cut) & (gdf["access_below_city_median"]),
            (gdf["hvi_rank"] >= medium_cut) & (gdf["access_below_city_median"]),
        ],
        ["High priority", "Medium priority"],
        default="Lower priority",
    )
    gdf["priority_sort"] = gdf["priority_class"].map({"High priority": 1, "Medium priority": 2, "Lower priority": 3})
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


def make_garden_pydeck(gardens: pd.DataFrame, buffer_miles: float = 0.25) -> pdk.Deck:
    data = gardens.copy()
    if len(data) == 0:
        data = pd.DataFrame([{"lon": NYC_CENTER["lon"], "lat": NYC_CENTER["lat"], "garden_name": "No gardens in filter", "borough": "", "status": ""}])
    radius_m = max(55, buffer_miles * 1609.34 * 0.12)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_radius=radius_m,
        get_fill_color=[14, 51, 39, 170],
        get_line_color=[255, 255, 255, 220],
        line_width_min_pixels=1,
        pickable=True,
        opacity=0.88,
    )
    view = pdk.ViewState(latitude=NYC_CENTER["lat"], longitude=NYC_CENTER["lon"], zoom=9.5, pitch=0)
    tooltip = {"html": "<b>{garden_name}</b><br/>{borough}<br/>{status}", "style": {"backgroundColor": "#0E3327", "color": "white"}}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="light")


def make_metric_choropleth(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    geojson = df_to_geojson(df)
    plot_df = df.copy()
    fig = px.choropleth_mapbox(
        plot_df,
        geojson=geojson,
        locations="nta_code",
        featureidkey="properties.nta_code",
        color=metric,
        hover_name="nta_name",
        hover_data={
            "borough": True,
            "garden_count": True,
            "population": ":,.0f",
            "garden_access_score": ":.2f",
            "garden_density_per_sq_mile": ":.2f",
            "hvi_rank": True if "hvi_rank" in plot_df.columns else False,
            "nta_code": False,
        },
        color_continuous_scale="YlGn",
        mapbox_style="carto-positron",
        center={"lat": NYC_CENTER["lat"], "lon": NYC_CENTER["lon"]},
        zoom=9.35,
        opacity=0.72,
        title=title,
    )
    fig.update_layout(
        height=640,
        margin={"r": 0, "t": 48, "l": 0, "b": 0},
        paper_bgcolor=PALETTE["cream"],
        plot_bgcolor=PALETTE["cream"],
        font={"color": PALETTE["ink"]},
        coloraxis_colorbar={"title": friendly_metric_name(metric)},
    )
    return fig


def make_priority_map(df: pd.DataFrame, gardens: pd.DataFrame) -> go.Figure:
    geojson = df_to_geojson(df)
    priority_colors = {"High priority": 3, "Medium priority": 2, "Lower priority": 1}
    plot_df = df.copy()
    plot_df["priority_numeric"] = plot_df["priority_class"].map(priority_colors)
    fig = px.choropleth_mapbox(
        plot_df,
        geojson=geojson,
        locations="nta_code",
        featureidkey="properties.nta_code",
        color="priority_numeric",
        hover_name="nta_name",
        hover_data={
            "borough": True,
            "priority_class": True,
            "hvi_rank": True,
            "garden_access_score": ":.2f",
            "garden_count": True,
            "nta_code": False,
            "priority_numeric": False,
        },
        color_continuous_scale=[
            [0.0, "#E8EFE2"],
            [0.49, "#E8EFE2"],
            [0.50, "#F4C999"],
            [0.74, "#F4C999"],
            [0.75, "#B85C38"],
            [1.0, "#B85C38"],
        ],
        mapbox_style="carto-positron",
        center={"lat": NYC_CENTER["lat"], "lon": NYC_CENTER["lon"]},
        zoom=9.35,
        opacity=0.74,
        title="Priority map: heat vulnerability + low garden access",
    )
    if len(gardens) > 0:
        fig.add_trace(
            go.Scattermapbox(
                lat=gardens["lat"],
                lon=gardens["lon"],
                mode="markers",
                marker={"size": 6, "color": PALETTE["forest"], "opacity": 0.72},
                text=gardens["garden_name"],
                name="Community gardens",
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )
    fig.update_layout(
        height=660,
        margin={"r": 0, "t": 48, "l": 0, "b": 0},
        paper_bgcolor=PALETTE["cream"],
        plot_bgcolor=PALETTE["cream"],
        font={"color": PALETTE["ink"]},
        coloraxis_showscale=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "xanchor": "left", "x": 0.01},
    )
    return fig


def render_data_factors() -> None:
    st.markdown("### What exactly goes into each dataset")
    st.dataframe(pd.DataFrame(DATASET_FACTORS), use_container_width=True, hide_index=True)
    st.markdown("#### Metric dictionary")
    st.dataframe(pd.DataFrame(METRIC_DICTIONARY_ROWS), use_container_width=True, hide_index=True)
    st.markdown("#### Heat Vulnerability Index factors")
    st.dataframe(pd.DataFrame(HVI_FACTOR_ROWS), use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="warning-callout">
        <strong>Important interpretation note:</strong> HVI is an index, not a direct medical outcome measure. 
        This dashboard uses it for exploratory priority mapping with community garden access, not for causal claims.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# App data pipeline
# -----------------------------------------------------------------------------

with st.spinner("Loading NYC garden, boundary, population, and heat-vulnerability data..."):
    raw_gardens, garden_load_status = load_greenthumb_data()
    nta_boundaries, nta_load_status = load_nta_boundaries()
    garden_points, garden_clean_status = clean_garden_points(raw_gardens)
    population_df, population_status = load_population_data(nta_boundaries)
    nta_metrics, garden_joined, metrics_status = compute_nta_metrics(nta_boundaries, garden_points, population_df)
    hvi_df, hvi_status = load_hvi_data()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.markdown("## Dashboard controls")
borough = st.sidebar.selectbox("Borough filter", get_borough_options(garden_points), index=0)
metric_view = st.sidebar.selectbox(
    "Metric view",
    ["garden_access_score", "garden_count", "garden_density_per_sq_mile"],
    format_func=friendly_metric_name,
)
buffer_distance = st.sidebar.slider("Visual garden buffer distance", 0.10, 1.00, 0.25, 0.05)
hvi_threshold = st.sidebar.slider("High HVI threshold", 3, 5, 4, 1)
render_selected_metric_explainer(metric_view, compact=True)

priority_gdf, priority_status = classify_priority_areas(nta_metrics, hvi_df, hvi_threshold=hvi_threshold)

filtered_gardens = filtered_by_borough(garden_points, borough)
filtered_nta = filtered_by_borough(priority_gdf, borough)
filtered_joined = filtered_by_borough(garden_joined, borough)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data quality")
mock_flags = [
    garden_load_status.get("mock") == "True",
    garden_clean_status.get("mock") == "True",
    nta_load_status.get("mock") == "True",
    population_status.get("mock") == "True" or metrics_status.get("population_mock") == "True",
    hvi_status.get("mock") == "True" or priority_status.get("hvi_mock") == "True",
]
st.sidebar.caption("🟢 Official/local data loaded" if not any(mock_flags) else "🟠 Some fallback/proxy data in use")
st.sidebar.caption(f"Gardens: {garden_load_status.get('source')}")
st.sidebar.caption(f"Population: {population_status.get('source')}")
st.sidebar.caption(f"HVI: {hvi_status.get('source')} / {priority_status.get('join_method')}")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-panel">
        <div class="eyebrow">UN/SDG education demo · Civic-tech geospatial dashboard · SDG 11 · SDG 13 · SDG 15</div>
        <div class="main-title">NYC Community Gardens for Equitable Climate Resilience</div>
        <div class="guiding-question">How can NYC community gardens support sustainable, equitable, and climate-resilient cities?</div>
        <p class="subtle" style="font-size:1rem; margin:.35rem 0 0 0;">
            This dashboard treats NYC community gardens as small-scale urban resilience infrastructure: public-space access, 
            heat-risk awareness, biodiversity, food education, and youth-led community action.
        </p>
        <div class="hero-meta">
            <span class="source-chip">NYC Open Data</span>
            <span class="source-chip">NTA geography</span>
            <span class="source-chip">Heat vulnerability</span>
            <span class="source-chip">Youth SDG action</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if any(mock_flags):
    st.markdown(
        """
        <div class="warning-callout">
        <strong>Data note:</strong> One or more layers are using fallback or proxy data. Use this as an education/prototype view, not a final policy estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1 · Overview",
        "2 · Access & Equity",
        "3 · Climate Resilience",
        "4 · SDG + Data Factors",
        "5 · Youth Action",
    ]
)

with tab1:
    st.subheader("NYC Community Gardens Overview")
    c1, c2, c3, c4 = st.columns(4)
    total_gardens = len(filtered_gardens)
    borough_count = filtered_gardens["borough"].nunique() if "borough" in filtered_gardens else 0
    active_share = np.nan
    if "status" in filtered_gardens and len(filtered_gardens) > 0:
        active_share = filtered_gardens["status"].astype(str).str.contains("active|license|open", case=False, regex=True).mean() * 100
    nta_with_gardens = int((filtered_nta["garden_count"] > 0).sum()) if len(filtered_nta) else 0
    with c1:
        render_kpi_card("Total gardens", f"{total_gardens:,}", "Filtered garden points")
    with c2:
        render_kpi_card("Boroughs represented", f"{borough_count:,}", "Current selection")
    with c3:
        render_kpi_card("Active/status share", "—" if pd.isna(active_share) else f"{active_share:.0f}%", "Based on available status text")
    with c4:
        render_kpi_card("NTAs with gardens", f"{nta_with_gardens:,}", "Neighborhoods containing ≥1 garden")

    st.pydeck_chart(make_garden_pydeck(filtered_gardens, buffer_miles=buffer_distance), use_container_width=True)
    st.markdown(
        """
        <div class="callout">
        <strong>Interpretation:</strong> Community gardens are distributed unevenly across NYC, creating an opportunity to examine access, equity, and climate resilience together.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(filtered_gardens) > 0:
        status_counts = filtered_gardens["status"].fillna("Unknown").astype(str).value_counts().head(12).reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(status_counts, x="status", y="count", title="Garden status split, if available")
        fig_status.update_layout(height=340, paper_bgcolor=PALETTE["cream"], plot_bgcolor=PALETTE["white"], font={"color": PALETTE["ink"]})
        st.plotly_chart(fig_status, use_container_width=True)

with tab2:
    st.subheader("Access & Equity Map")
    st.markdown(
        """
        <span class="metric-pill">Garden Count by NTA</span>
        <span class="metric-pill">Garden Access Score = gardens per 10,000 residents</span>
        <span class="metric-pill">Garden Density = gardens per square mile</span>
        """,
        unsafe_allow_html=True,
    )
    render_selected_metric_explainer(metric_view)
    st.plotly_chart(
        make_metric_choropleth(filtered_nta, metric_view, f"NTA choropleth: {friendly_metric_name(metric_view)}"),
        use_container_width=True,
    )
    st.markdown(
        """
        <div class="warning-callout">
        <strong>Caveat:</strong> Population denominators and boundary definitions affect interpretation. A low access score can mean few gardens, high population, or both.
        </div>
        """,
        unsafe_allow_html=True,
    )
    bottom = filtered_nta.sort_values(["garden_access_score", "garden_count"], ascending=[True, True]).head(10)
    st.markdown("#### Bottom 10 NTAs by garden access score")
    st.dataframe(
        bottom[["nta_code", "nta_name", "borough", "population", "garden_count", "garden_access_score", "garden_density_per_sq_mile"]].rename(
            columns={
                "nta_code": "NTA Code",
                "nta_name": "NTA Name",
                "borough": "Borough",
                "population": "Population",
                "garden_count": "Garden Count",
                "garden_access_score": "Gardens per 10k Residents",
                "garden_density_per_sq_mile": "Gardens per Sq Mile",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.subheader("Climate Resilience Layer")
    st.markdown(
        """
        <div class="warning-callout">
        <strong>Exploratory Analysis/Priority Mapping Only:</strong> This matrix explores structural correlations, not direct causal health outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_selected_metric_explainer("hvi_rank")
    render_selected_metric_explainer("priority_class")
    st.plotly_chart(make_priority_map(filtered_nta, filtered_gardens), use_container_width=True)
    high = filtered_nta[filtered_nta["priority_class"] == "High priority"]
    medium = filtered_nta[filtered_nta["priority_class"] == "Medium priority"]
    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("High priority NTAs", f"{len(high):,}", f"HVI ≥ {hvi_threshold} + below-median access")
    with col2:
        render_kpi_card("Medium priority NTAs", f"{len(medium):,}", "Moderate/high HVI + below-median access")
    with col3:
        render_kpi_card("Median access score", format_num(city_median(filtered_nta["garden_access_score"]), 2), "Gardens per 10,000 residents")
    st.markdown(
        """
        <div class="callout">
        <strong>Insight:</strong> Neighborhoods with high heat vulnerability and low community garden access represent priority areas for youth-led sustainability education.
        </div>
        """,
        unsafe_allow_html=True,
    )
    priority_table = filtered_nta.sort_values(["priority_sort", "hvi_rank", "garden_access_score"], ascending=[True, False, True])
    st.dataframe(
        priority_table[["nta_code", "nta_name", "borough", "priority_class", "hvi_rank", "garden_access_score", "garden_count", "population"]].rename(
            columns={
                "nta_code": "NTA Code",
                "nta_name": "NTA Name",
                "borough": "Borough",
                "priority_class": "Priority",
                "hvi_rank": "HVI Rank",
                "garden_access_score": "Gardens per 10k Residents",
                "garden_count": "Garden Count",
                "population": "Population",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.subheader("SDG Lens + Data Factor Transparency")
    sdgs = [
        {"sdg": "SDG 11", "title": "Sustainable Cities", "badge": "Inclusive public space", "body": "Gardens can improve neighborhood access to safe, local, community-managed green space."},
        {"sdg": "SDG 13", "title": "Climate Action", "badge": "Heat adaptation", "body": "Garden maps help students connect heat vulnerability to local adaptation and awareness."},
        {"sdg": "SDG 15", "title": "Life on Land", "badge": "Urban biodiversity", "body": "Community gardens can support pollinators, habitat, soil learning, and ecological stewardship."},
        {"sdg": "SDG 2", "title": "Zero Hunger", "badge": "Food education", "body": "Gardens can teach food systems, nutrition, and community-based food security."},
        {"sdg": "SDG 3", "title": "Good Health", "badge": "Wellness and learning", "body": "Outdoor learning and community care can support wellness, belonging, and health awareness."},
    ]
    cols = st.columns(3)
    for i, item in enumerate(sdgs):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="sdg-card">
                    <span class="badge">{item['sdg']}</span>
                    <h4 style="color:{PALETTE['forest']}; margin:.6rem 0 .25rem 0;">{item['title']}</h4>
                    <strong style="color:{PALETTE['terracotta']};">{item['badge']}</strong>
                    <p class="subtle" style="margin-top:.55rem;">{item['body']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("---")
    render_data_factors()

with tab5:
    st.subheader("Youth Action & Recommendations")
    priority_for_actions = filtered_nta[filtered_nta["priority_class"].isin(["High priority", "Medium priority"])].sort_values(
        ["priority_sort", "hvi_rank", "garden_access_score"], ascending=[True, False, True]
    )
    top_names = priority_for_actions["nta_name"].head(5).tolist()
    adopt_text = ", ".join(top_names) if top_names else "No priority neighborhoods in the current filter"
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"""
            <div class="action-card">
            <span class="badge">1</span>
            <h4>What students can learn</h4>
            <p>How open data connects public space, population, heat vulnerability, and SDG action. Students can compare garden access across neighborhoods and ask why patterns differ.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="action-card">
            <span class="badge">2</span>
            <h4>Priority neighborhoods to adopt</h4>
            <p>{adopt_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(
            """
            <div class="action-card">
            <span class="badge">3</span>
            <h4>SDG action ideas</h4>
            <p>Run a garden-access walk audit, design heat-safety posters, map shade and cooling resources, interview gardeners, and create a youth SDG story map.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            """
            <div class="action-card">
            <span class="badge">4</span>
            <h4>Community partnership ideas</h4>
            <p>Partner with GreenThumb garden groups, local schools, libraries, senior centers, youth climate clubs, and neighborhood health organizations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    download_cols = ["nta_code", "nta_name", "borough", "priority_class", "hvi_rank", "garden_access_score", "garden_count", "population"]
    csv = priority_for_actions[download_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download priority neighborhoods CSV",
        csv,
        file_name="nyc_garden_heat_priority_neighborhoods.csv",
        mime="text/csv",
    )

    st.markdown("#### 60-second presenter script")
    st.text_area(
        "Presenter script",
        value=(
            "Hello, my dashboard asks one question: How can NYC community gardens support sustainable, equitable, and climate-resilient cities? "
            "I treat community gardens as small-scale urban resilience infrastructure, not just as nice green spaces. "
            "First, I map where gardens are located across New York City. Then I connect those garden points to neighborhood boundaries and population data to calculate access: gardens per 10,000 residents. "
            "Next, I compare access with heat vulnerability. Neighborhoods with high heat vulnerability and low garden access become priority areas for youth-led SDG education. "
            "This is not a causal health study; it is an exploratory priority map. The goal is to help students ask better civic questions, identify neighborhoods for partnership, and connect local action to SDG 11, SDG 13, SDG 15, SDG 2, and SDG 3."
        ),
        height=190,
    )

# -----------------------------------------------------------------------------
# Footer: caveats and sources
# -----------------------------------------------------------------------------

with st.expander("Data caveats, source notes, and loading status", expanded=False):
    st.markdown(
        """
        **Core caveats**
        - This is an exploratory civic-tech dashboard, not a causal health-outcome study.
        - Garden access is based on point-in-polygon assignment to NTAs, not walking network travel time.
        - Population denominator choice changes the access score.
        - HVI may use a CDTA-to-NTA proxy when a direct NTA-level HVI table is unavailable.
        - Garden status, entrance availability, hours, program quality, and capacity are not fully represented.
        """
    )
    status_df = pd.DataFrame(
        [
            garden_load_status,
            {"dataset": "Garden point cleaning", "dataset_id": "derived", "source": "app logic", "mock": garden_clean_status.get("mock"), "message": garden_clean_status.get("message")},
            nta_load_status,
            population_status,
            hvi_status,
            {"dataset": "Priority classification", "dataset_id": "derived", "source": priority_status.get("join_method"), "mock": priority_status.get("hvi_mock"), "message": priority_status.get("message")},
        ]
    )
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    render_data_factors()
