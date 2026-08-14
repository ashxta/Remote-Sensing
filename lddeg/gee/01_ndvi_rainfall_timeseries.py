"""
Export annual post-monsoon NDVI composites (1990-2025) + annual rainfall
for Karbi Anglong, Assam. Landsat 5/7/8/9 Collection-2 Level-2, harmonized.

Post-monsoon window (Oct 15 - Dec 31) is used because it is the least
cloudy season in Northeast India and captures peak-biomass conditions.

Run after `earthengine authenticate`:
    python gee/01_ndvi_rainfall_timeseries.py
Outputs to Drive folder 'lddeg_karbi':
    ndvi_stack.tif  (36 bands, one per year, 30 m)
    rain_stack.tif  (36 bands, annual CHIRPS totals, resampled to 30 m grid)
"""
import ee

ee.Initialize()

# Karbi Anglong approximate bounding box (trim with district shapefile later)
AOI = ee.Geometry.Rectangle([92.30, 25.55, 93.85, 26.60])
YEARS = list(range(1990, 2026))
SCALE = 30


def mask_l2(img):
    qa = img.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4)) \
              .Or(qa.bitwiseAnd(1 << 1))          # cloud, shadow, dilated
    sat = img.select("QA_RADSAT").neq(0)
    return img.updateMask(cloud.Not()).updateMask(sat.Not())


def scale_sr(img):
    return img.multiply(0.0000275).add(-0.2)


def ndvi_tm_etm(img):                              # Landsat 4/5/7
    img = mask_l2(img)
    red = scale_sr(img.select("SR_B3"))
    nir = scale_sr(img.select("SR_B4"))
    return nir.subtract(red).divide(nir.add(red)).rename("ndvi") \
              .copyProperties(img, ["system:time_start"])


def ndvi_oli(img):                                 # Landsat 8/9
    img = mask_l2(img)
    red = scale_sr(img.select("SR_B4"))
    nir = scale_sr(img.select("SR_B5"))
    ndvi = nir.subtract(red).divide(nir.add(red))
    # Roy et al. (2016) OLI->ETM+ NDVI harmonization
    ndvi = ndvi.multiply(0.9589).add(0.0029)
    return ndvi.rename("ndvi").copyProperties(img, ["system:time_start"])


l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(ndvi_tm_etm)
l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(ndvi_tm_etm)
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(ndvi_oli)
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(ndvi_oli)
landsat = l5.merge(l7).merge(l8).merge(l9).filterBounds(AOI)

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(AOI)


def annual_ndvi(y):
    y = ee.Number(y)
    start = ee.Date.fromYMD(y, 10, 15)
    end = ee.Date.fromYMD(y, 12, 31)
    comp = landsat.filterDate(start, end).median()
    return comp.rename(ee.String("ndvi_").cat(y.format("%d")))


def annual_rain(y):
    y = ee.Number(y)
    # hydrological year Jan-Dec preceding the post-monsoon composite
    total = chirps.filterDate(ee.Date.fromYMD(y, 1, 1),
                              ee.Date.fromYMD(y, 12, 31)).sum()
    return total.rename(ee.String("rain_").cat(y.format("%d")))


ndvi_stack = ee.ImageCollection(
    [annual_ndvi(y) for y in YEARS]).toBands().clip(AOI).toFloat()
rain_stack = ee.ImageCollection(
    [annual_rain(y) for y in YEARS]).toBands().clip(AOI).toFloat()

ee.batch.Export.image.toDrive(
    image=ndvi_stack, description="ndvi_stack", folder="lddeg_karbi",
    region=AOI, scale=SCALE, maxPixels=1e10).start()
ee.batch.Export.image.toDrive(
    image=rain_stack, description="rain_stack", folder="lddeg_karbi",
    region=AOI, scale=SCALE, maxPixels=1e10).start()

print("Exports started: ndvi_stack (36 bands), rain_stack (36 bands).")
print("Note: Landsat-7 SLC-off gaps after May 2003 are filled by the")
print("median compositing where L5/L8 overlap exists; residual gaps")
print("remain 2003-2012 and are handled as NaN by the pipeline.")
