"""
SYNTHETIC demo dataset: a Karbi Anglong-shaped landscape with 36 years
(1990-2025) of annual post-monsoon NDVI + rainfall.

Encoded geography (schematic): hill mass in the centre-east (forest +
jhum belts on mid-slopes), plains in the west/north (cropland),
degradation hotspots near settlement/quarry patches, recovery patches
(plantation/regeneration), and rainfall with wet/dry years driving
everything so that RESTREND has real work to do.

Classes (truth raster). These names describe what the GENERATOR PLANTED,
not anything detected or verified. "jhum" here means "the archetype this
script builds to imitate a rotational-cultivation signal"; nothing in the
analysis pipeline may conclude that a cyclic signal is jhum, and the
analytical trajectory classes deliberately avoid the term:
  1 stable forest   2 cropland   3 jhum (cyclic, period 5-11 yr,
  shortening over time)   4 degrading (monotonic decline after a break
  year)   5 recovering (increase after a break year)

REPLACE with GEE exports (ndvi_stack.tif, rain_stack.tif) for the real
study; the pipeline is unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

rng = np.random.default_rng(C.SEED)
H, W = 200, 300
T = len(C.YEARS)
transform = from_origin(92.30, 26.60, 1.55 / W, 1.05 / H)
CRS = "EPSG:4326"

yy, xx = np.mgrid[0:H, 0:W] / np.array([[H], [W]]).reshape(2, 1, 1)


def smooth(scale, sigma):
    return gaussian_filter(rng.normal(0, scale, (H, W)), sigma)


# ---------------- landscape zones -------------------------------------------
hill = np.exp(-(((yy - 0.55) ** 2) / 0.09 + ((xx - 0.60) ** 2) / 0.10)) \
       + 0.5 * np.exp(-(((yy - 0.30) ** 2) / 0.02 + ((xx - 0.80) ** 2) / 0.02))
hill = gaussian_filter(hill, 4) + smooth(0.05, 8)
elev = 60 + 900 * np.clip(hill, 0, None)

truth = np.full((H, W), 2, "int16")                       # default cropland
truth[hill > 0.45] = 1                                    # core forest
jhum_belt = (hill > 0.18) & (hill <= 0.45) & (smooth(1, 12) > -0.3)
truth[jhum_belt] = 3

# degradation hotspots: fringe forest near plains (encroachment/quarry)
for _ in range(14):
    cy, cx = rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.9)
    r = rng.uniform(0.015, 0.045)
    blob = ((yy - cy) ** 2 + ((xx - cx) * 1.5) ** 2) < r ** 2
    truth[blob & (truth != 2)] = 4
# recovery patches (plantation / fallow regeneration)
for _ in range(10):
    cy, cx = rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.9)
    r = rng.uniform(0.012, 0.035)
    blob = ((yy - cy) ** 2 + ((xx - cx) * 1.5) ** 2) < r ** 2
    truth[blob & (truth == 3)] = 5
    truth[blob & (truth == 4)] = 5

# ---------------- rainfall series -------------------------------------------
t = np.arange(T)
regional = (1800 + 250 * np.sin(2 * np.pi * t / 7.3)
            + rng.normal(0, 180, T))                       # wet/dry cycles
regional[[9, 19, 30]] -= 420                               # drought years
rain = (regional[:, None, None]
        * (1 + 0.25 * (yy[None] - 0.5))                    # N-S gradient
        + gaussian_filter(rng.normal(0, 60, (T, H, W)), (0, 6, 6)))
rain = rain.astype("float32")

rain_anom = (rain - rain.mean(0)) / (rain.std(0) + 1e-6)

# ---------------- NDVI generation per class ---------------------------------
ndvi = np.zeros((T, H, W), "float32")
base_noise = gaussian_filter(rng.normal(0, 0.02, (T, H, W)), (0, 2, 2))

# 1 stable forest: high, weak rain response
m = truth == 1
n1 = m.sum()
base1 = rng.uniform(0.68, 0.85, n1)
ndvi[:, m] = base1[None, :] + 0.02 * rain_anom[:, m] \
    + 0.030 * rng.normal(size=(T, n1))

# 2 cropland: moderate, strong rain response, tiny greening trend
m = truth == 2
n2 = m.sum()
base2 = rng.uniform(0.36, 0.56, n2)
trend2 = rng.uniform(-0.0006, 0.0018, n2)          # mixed mild trends
ndvi[:, m] = (base2[None, :] + trend2[None, :] * t[:, None]
              + 0.07 * rain_anom[:, m]
              + 0.040 * rng.normal(size=(T, n2)))

# 3 jhum: sawtooth cycles, period shortening 11 -> 5 yr AND (for ~65% of
# pixels) an intensification envelope: shorter fallow = lower peak regrowth,
# so the cycle rides on a slow decline. This is the real NE-India story and
# is exactly what makes linear trend tests mislabel cyclic land.
m = truth == 3
n3 = m.sum()
phase = rng.uniform(0, 1, n3)
period0 = rng.uniform(9, 12, n3)
period1 = rng.uniform(4.5, 6.5, n3)
intensifying = rng.random(n3) < 0.65
peak_drop = np.where(intensifying, rng.uniform(0.12, 0.32, n3), 0.0)
base3 = rng.uniform(0.40, 0.55, n3)
for i, ti in enumerate(t):
    frac = ti / (T - 1)
    per = period0 * (1 - frac) + period1 * frac
    cyc = ((ti / per + phase) % 1.0)
    envelope = 1.0 - peak_drop * frac / 0.75
    val = np.where(cyc < 0.15, base3 - 0.18,
                   base3 - 0.18 + (0.42 * envelope) * (cyc - 0.15) / 0.85)
    ndvi[i, m] = val + 0.03 * rain_anom[i, m] \
        + 0.035 * rng.normal(size=n3)

# 4 degrading: stable until a break year, then monotonic loss
m = truth == 4
n4 = m.sum()
bk = rng.integers(6, T - 8, n4)
rate = rng.uniform(0.010, 0.022, n4)
start4 = rng.uniform(0.55, 0.75, n4)
for i, ti in enumerate(t):
    after = np.clip(ti - bk, 0, None)
    ndvi[i, m] = (start4 - rate * after + 0.03 * rain_anom[i, m]
                  + 0.035 * rng.normal(size=n4)).clip(0.12, 0.9)

# 5 recovering: low until a break year, then gain
m = truth == 5
n5 = m.sum()
bk = rng.integers(6, T - 8, n5)
rate = rng.uniform(0.012, 0.024, n5)
start5 = rng.uniform(0.25, 0.42, n5)
for i, ti in enumerate(t):
    after = np.clip(ti - bk, 0, None)
    ndvi[i, m] = (start5 + rate * after + 0.03 * rain_anom[i, m]
                  + 0.035 * rng.normal(size=n5)).clip(0.1, 0.85)

ndvi += base_noise
# mixed pixels: mild spatial blending across class boundaries (30 m reality)
from scipy.ndimage import gaussian_filter as _gf
for i in range(T):
    ndvi[i] = 0.75 * ndvi[i] + 0.25 * _gf(ndvi[i], 1.2)
ndvi += rng.normal(0, 0.012, ndvi.shape).astype("float32")
ndvi = ndvi.clip(0.05, 0.95)

# simulate Landsat-7 SLC-off era sparse gaps (2004-2012) as NaN speckle
for yi, yr in enumerate(C.YEARS):
    if 2004 <= yr <= 2012:
        gapmask = rng.random((H, W)) < 0.04
        ndvi[yi][gapmask] = np.nan
# pipeline requirement: fill NaN by temporal linear interpolation
for _ in range(2):
    nanm = np.isnan(ndvi)
    if not nanm.any():
        break
    filled = np.where(nanm, np.roll(ndvi, 1, axis=0), ndvi)
    filled = np.where(np.isnan(filled), np.roll(ndvi, -1, axis=0), filled)
    ndvi = filled
ndvi = np.nan_to_num(ndvi, nan=0.45)

# ---------------- write rasters ---------------------------------------------
C.RASTERS.mkdir(parents=True, exist_ok=True)
meta = dict(driver="GTiff", height=H, width=W, count=T, dtype="float32",
            crs=CRS, transform=transform)
with rasterio.open(C.NDVI_STACK, "w", **meta) as dst:
    for i, yr in enumerate(C.YEARS):
        dst.write(ndvi[i], i + 1)
        dst.set_band_description(i + 1, f"ndvi_{yr}")
with rasterio.open(C.RAIN_STACK, "w", **meta) as dst:
    for i, yr in enumerate(C.YEARS):
        dst.write(rain[i], i + 1)
        dst.set_band_description(i + 1, f"rain_{yr}")
with rasterio.open(C.TRUTH, "w", **{**meta, "count": 1,
                                    "dtype": "int16"}) as dst:
    dst.write(truth, 1)

counts = {C.CLASSES[k]: int((truth == k).sum()) for k in sorted(C.CLASSES)}
print(f"Synthetic stacks written: {T} years, {H}x{W}")
print("Class pixel counts:", counts)
