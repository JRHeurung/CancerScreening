# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Explore PLACES CRC screening — New York ZCTAs
# MAGIC
# MAGIC **Goal:** profile the PLACES ZCTA table, keep New York, save a clean silver table, and map CRC screening by ZIP area.
# MAGIC
# MAGIC | Step | Output |
# MAGIC |---|---|
# MAGIC | 1. Load bronze | `workspace.crc.bronze_places_zcta` → pandas |
# MAGIC | 2. Clean + filter NY | typed columns, CI split, lat/lon from `Geolocation` |
# MAGIC | 3. Profile | counts, distribution, extremes, CI width vs population |
# MAGIC | 4. Save silver | `workspace.crc.silver_places_zcta_ny` |
# MAGIC | 5. Map A (no extra data) | dot map at each ZCTA centroid |
# MAGIC | 6. Map B (needs boundary file) | choropleth of ZCTA polygons, statewide + NYC zoom |
# MAGIC
# MAGIC **Before Map B:** upload the Census ZCTA boundary zip to the volume (see the cell above Map B).

# COMMAND ----------

# MAGIC %pip install -q "plotly>=5.24" geopandas pyogrio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# --- Settings: change these if your names differ --------------------------------
CATALOG_SCHEMA = "workspace.crc"
BRONZE_TABLE   = f"{CATALOG_SCHEMA}.bronze_places_zcta"
SILVER_TABLE   = f"{CATALOG_SCHEMA}.silver_places_zcta_ny"
VOLUME_DIR     = "/Volumes/workspace/crc/raw"
ZCTA_SHAPE_ZIP = f"{VOLUME_DIR}/geo/cb_2020_us_zcta520_500k.zip"

# Measure prefix in PLACES. COLON_SCREEN = CRC screening, adults 45–75.
MEASURE = "colon_screen"

# Sequential single-hue ramp (light = low, dark = high).
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# COMMAND ----------

# MAGIC %md ## 1 · Load bronze and look at the columns

# COMMAND ----------

import re
import numpy as np
import pandas as pd

bronze = spark.table(BRONZE_TABLE)
print(f"{bronze.count():,} rows, {len(bronze.columns)} columns")

# Column names vary in case between downloads; map lowercase -> actual.
colmap = {c.lower(): c for c in bronze.columns}
print("Columns:", sorted(colmap))

# COMMAND ----------

def pick(*candidates):
    """Return the real column name for the first candidate that exists (case-insensitive)."""
    for c in candidates:
        if c.lower() in colmap:
            return colmap[c.lower()]
    return None

col_zcta   = pick("zcta5", "locationname", "zcta")
col_pop    = pick("totalpopulation")
col_pop18  = pick("totalpop18plus")
col_rate   = pick(f"{MEASURE}_crudeprev")
col_ci     = pick(f"{MEASURE}_crude95ci")
col_geo    = pick("geolocation")

assert col_zcta and col_rate, (
    "Could not find the ZCTA or CRC screening column. "
    "Check the printed column list and edit pick(...) above."
)
print(dict(zcta=col_zcta, pop=col_pop, pop18=col_pop18, rate=col_rate, ci=col_ci, geo=col_geo))

keep = [c for c in [col_zcta, col_pop, col_pop18, col_rate, col_ci, col_geo] if c]
raw = bronze.select(*keep).toPandas()
raw.head()

# COMMAND ----------

# MAGIC %md ## 2 · Clean and filter to New York
# MAGIC
# MAGIC The PLACES ZCTA file has no state column. For now we use the agreed ZIP-prefix rule
# MAGIC (**100–149**, plus `06390` Fishers Island). Swap this for the Census ZCTA-to-county crosswalk later.

# COMMAND ----------

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def parse_ci(s):
    """'( 60.1, 70.2)' -> (60.1, 70.2)"""
    nums = s.astype(str).str.findall(r"-?\d+(?:\.\d+)?")
    lo = nums.str[0].astype(float)
    hi = nums.str[1].astype(float)
    return lo, hi

def parse_point(s):
    """'POINT (-73.95 40.78)' -> (lon, lat)"""
    nums = s.astype(str).str.findall(r"-?\d+(?:\.\d+)?")
    lon = nums.str[0].astype(float)
    lat = nums.str[1].astype(float)
    return lon, lat

def is_ny_zcta(z):
    if z == "06390":
        return True
    return z[:3].isdigit() and 100 <= int(z[:3]) <= 149

df = pd.DataFrame()
df["zcta"] = raw[col_zcta].astype(str).str.strip().str.zfill(5)   # keep as TEXT, keep leading zeros
df["crc_screen_pct"] = to_num(raw[col_rate])
if col_pop:   df["total_pop"]  = to_num(raw[col_pop]).astype("Int64")
if col_pop18: df["adult_pop"]  = to_num(raw[col_pop18]).astype("Int64")
if col_ci:    df["ci_low"], df["ci_high"] = parse_ci(raw[col_ci])
if col_geo:   df["lon"], df["lat"] = parse_point(raw[col_geo])

df["state"] = np.where(df["zcta"].map(is_ny_zcta), "NY", None)
ny = df[df["state"] == "NY"].dropna(subset=["crc_screen_pct"]).copy()

if col_ci:
    ny["ci_width"] = ny["ci_high"] - ny["ci_low"]

print(f"NY ZCTAs with a CRC estimate: {len(ny):,} (of {len(df):,} national)")
ny.head()

# COMMAND ----------

# MAGIC %md ## 3 · Profile

# COMMAND ----------

summary = ny["crc_screen_pct"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1)
display(summary.to_frame("crc_screen_pct").reset_index())

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go

fig = px.histogram(
    ny, x="crc_screen_pct", nbins=40,
    color_discrete_sequence=[BLUES[4]],
    labels={"crc_screen_pct": "Adults 45–75 up to date on CRC screening (%)"},
    title="Distribution of CRC screening across New York ZCTAs",
)
fig.update_traces(marker_line_color="white", marker_line_width=1,
                  hovertemplate="%{x}%<br>%{y} ZCTAs<extra></extra>")
fig.update_layout(yaxis_title="Number of ZCTAs", bargap=0.05, template="plotly_white")
fig.show()

# COMMAND ----------

# Lowest and highest 10 (only areas with a meaningful population)
cols = [c for c in ["zcta", "crc_screen_pct", "ci_low", "ci_high", "total_pop"] if c in ny]
big = ny[ny.get("total_pop", pd.Series(1000, index=ny.index)).fillna(0) >= 1000]
print("Lowest 10:");  display(big.nsmallest(10, "crc_screen_pct")[cols])
print("Highest 10:"); display(big.nlargest(10, "crc_screen_pct")[cols])

# COMMAND ----------

# How uncertain are small areas? CI width vs population.
if "ci_width" in ny and "total_pop" in ny:
    fig = px.scatter(
        ny, x="total_pop", y="ci_width", log_x=True,
        color_discrete_sequence=[BLUES[4]], opacity=0.6,
        hover_data={"zcta": True, "crc_screen_pct": ":.1f"},
        labels={"total_pop": "ZCTA population (log scale)", "ci_width": "95% CI width (percentage points)"},
        title="Smaller ZCTAs have wider confidence intervals",
    )
    fig.update_traces(marker_size=8, marker_line_color="white", marker_line_width=1)
    fig.update_layout(template="plotly_white")
    fig.show()

# COMMAND ----------

# MAGIC %md ## 4 · Save silver table

# COMMAND ----------

(spark.createDataFrame(ny.astype({c: "float" for c in ["total_pop", "adult_pop"] if c in ny}))
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(SILVER_TABLE))
print(f"Saved {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Map A — dot map at ZCTA centroids (no extra data needed)
# MAGIC Each dot is one ZIP area, colored by screening rate. Quick look before the polygon map.

# COMMAND ----------

color_range = [np.floor(ny["crc_screen_pct"].quantile(0.02)), np.ceil(ny["crc_screen_pct"].quantile(0.98))]

hover = {"zcta": True, "crc_screen_pct": ":.1f", "lat": False, "lon": False}
if "ci_low" in ny:    hover.update(ci_low=":.1f", ci_high=":.1f")
if "total_pop" in ny: hover.update(total_pop=":,")

if {"lat", "lon"} <= set(ny.columns):
    fig = px.scatter_map(
        ny, lat="lat", lon="lon", color="crc_screen_pct",
        color_continuous_scale=BLUES, range_color=color_range,
        hover_data=hover,
        labels={"crc_screen_pct": "Screened %", "ci_low": "CI low", "ci_high": "CI high", "total_pop": "Population"},
        center={"lat": 42.9, "lon": -75.5}, zoom=5.6, height=650,
        map_style="carto-positron",
        title="CRC screening, adults 45–75, by NY ZCTA (CDC PLACES)",
    )
    fig.update_traces(marker_size=7)
    fig.show()
else:
    print("No Geolocation column; skip to Map B.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Map B — choropleth of ZCTA boundaries
# MAGIC
# MAGIC **One-time setup on your laptop (PowerShell):** download the Census 2020 ZCTA cartographic boundary file (~60 MB) and copy it to the volume.
# MAGIC
# MAGIC ```powershell
# MAGIC Invoke-WebRequest https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip -OutFile data\raw\cb_2020_us_zcta520_500k.zip
# MAGIC databricks fs mkdir dbfs:/Volumes/workspace/crc/raw/geo
# MAGIC databricks fs cp data\raw\cb_2020_us_zcta520_500k.zip dbfs:/Volumes/workspace/crc/raw/geo/
# MAGIC ```
# MAGIC Make sure `data/raw/` is in `.gitignore` so the zip never goes to GitHub.

# COMMAND ----------

import geopandas as gpd

shapes = gpd.read_file(ZCTA_SHAPE_ZIP)
shapes["zcta"] = shapes["ZCTA5CE20"].astype(str).str.zfill(5)
ny_shapes = shapes[shapes["zcta"].isin(ny["zcta"])].copy()

# Simplify geometry so the map loads fast (~100 m tolerance in degrees).
ny_shapes["geometry"] = ny_shapes.to_crs(4326).geometry.simplify(0.001, preserve_topology=True)
geojson = ny_shapes[["zcta", "geometry"]].__geo_interface__

missing = set(ny["zcta"]) - set(ny_shapes["zcta"])
print(f"{len(ny_shapes):,} NY ZCTA shapes; {len(missing)} PLACES ZCTAs without a shape")

# COMMAND ----------

def choropleth(center, zoom, title):
    fig = px.choropleth_map(
        ny, geojson=geojson, locations="zcta", featureidkey="properties.zcta",
        color="crc_screen_pct", color_continuous_scale=BLUES, range_color=color_range,
        hover_data={k: v for k, v in hover.items() if k not in ("lat", "lon")},
        labels={"crc_screen_pct": "Screened %", "ci_low": "CI low", "ci_high": "CI high", "total_pop": "Population"},
        center=center, zoom=zoom, opacity=0.85, height=700,
        map_style="carto-positron", title=title,
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),
                      coloraxis_colorbar=dict(title="Screened %", ticksuffix="%"))
    return fig

fig_state = choropleth({"lat": 42.9, "lon": -75.5}, 5.6, "CRC screening, adults 45–75, by NY ZCTA (CDC PLACES)")
fig_state.show()

# COMMAND ----------

fig_nyc = choropleth({"lat": 40.70, "lon": -73.95}, 9.3, "CRC screening by ZCTA — New York City")
fig_nyc.show()

# COMMAND ----------

# Optional: save the statewide map as a standalone HTML you can open or share.
out = f"{VOLUME_DIR}/maps/ny_crc_screening_zcta.html"
dbutils.fs.mkdirs(f"{VOLUME_DIR}/maps")
fig_state.write_html(out, include_plotlyjs="cdn")
print("Saved", out)
