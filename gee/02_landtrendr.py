"""
Run GEE's native LandTrendr temporal segmentation on the annual NDVI
series and export disturbance year + magnitude for Karbi Anglong.

    python gee/02_landtrendr.py
Outputs to Drive folder 'lddeg_karbi':
    lt_disturbance.tif  (band 1: year of greatest NDVI loss,
                         band 2: magnitude of that loss,
                         band 3: duration of the loss segment)
"""
import ee

ee.Initialize()

AOI = ee.Geometry.Rectangle([92.30, 25.55, 93.85, 26.60])
YEARS = list(range(1990, 2026))


def mask_l2(img):
    qa = img.select("QA_PIXEL")
    bad = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4)) \
            .Or(qa.bitwiseAnd(1 << 1))
    return img.updateMask(bad.Not())


def annual_img(y):
    y = ee.Number(y)
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    def nd_tm(i):
        i = mask_l2(i)
        r = i.select("SR_B3").multiply(0.0000275).add(-0.2)
        n = i.select("SR_B4").multiply(0.0000275).add(-0.2)
        return n.subtract(r).divide(n.add(r))

    def nd_oli(i):
        i = mask_l2(i)
        r = i.select("SR_B4").multiply(0.0000275).add(-0.2)
        n = i.select("SR_B5").multiply(0.0000275).add(-0.2)
        return n.subtract(r).divide(n.add(r)).multiply(0.9589).add(0.0029)

    coll = l5.map(nd_tm).merge(l7.map(nd_tm)).merge(l8.map(nd_oli)) \
             .filterBounds(AOI) \
             .filterDate(ee.Date.fromYMD(y, 10, 15),
                         ee.Date.fromYMD(y, 12, 31))
    # LandTrendr expects LOSS as positive change on band 1 -> invert NDVI
    return coll.median().multiply(-1).rename("ndvi_inv") \
        .set("system:time_start", ee.Date.fromYMD(y, 11, 15).millis())


series = ee.ImageCollection([annual_img(y) for y in YEARS])

lt = ee.Algorithms.TemporalSegmentation.LandTrendr(
    timeSeries=series,
    maxSegments=6,
    spikeThreshold=0.9,
    vertexCountOvershoot=3,
    preventOneYearRecovery=True,
    recoveryThreshold=0.5,
    pvalThreshold=0.05,
    bestModelProportion=0.75,
    minObservationsNeeded=8,
)

# Unpack the segmentation array: rows = [year, source, fitted, isVertex]
ltarr = lt.select("LandTrendr")
years_img = ltarr.arraySlice(0, 0, 1)
fitted = ltarr.arraySlice(0, 2, 3).multiply(-1)     # back to real NDVI
vertex = ltarr.arraySlice(0, 3, 4)

# Segment deltas between consecutive vertices
vyears = years_img.arrayMask(vertex)
vfit = fitted.arrayMask(vertex)
n = vfit.arrayLength(1)
d_fit = vfit.arraySlice(1, 1).subtract(vfit.arraySlice(1, 0, -1))
d_yr = vyears.arraySlice(1, 1)

# Greatest loss segment
loss_mag = d_fit.multiply(-1)                       # positive = NDVI loss
max_loss = loss_mag.arrayReduce(ee.Reducer.max(), [1])
idx = loss_mag.arrayArgmax().arrayFlatten([["i"], ["j"]]).select(1)
loss_year = d_yr.arrayGet(ee.Image.constant(0).addBands(idx))
seg_len = d_yr.subtract(vyears.arraySlice(1, 0, -1)) \
              .arrayGet(ee.Image.constant(0).addBands(idx))

out = (loss_year.rename("loss_year")
       .addBands(max_loss.arrayFlatten([["m"], ["v"]]).rename("loss_mag"))
       .addBands(seg_len.rename("loss_duration"))
       .toFloat().clip(AOI))

ee.batch.Export.image.toDrive(
    image=out, description="lt_disturbance", folder="lddeg_karbi",
    region=AOI, scale=30, maxPixels=1e10).start()

print("LandTrendr export started.")
