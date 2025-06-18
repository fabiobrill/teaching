import os
import geopandas as gpd
import rioxarray
import rasterio
from rasterio.features import rasterize

os.getcwd()

#datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data"
datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/testset"
os.listdir(datafolder)

# load the file "/HWMI_2011_2020_present_30.tif" as reference_raster
reference_raster = rioxarray.open_rasterio(datafolder + "/HWMI_2011_2020_present_30.tif")
reference_raster.plot()

# load the file bbox_simulation.gpkg and restrict the extent of the rasters to this bbox
bbox_subset = gpd.read_file(datafolder + "/prediction-subset/subset-bbox.gpkg")
# Uncomment the following line if you want to load a specific city file
#berlin = gpd.read_file(datafolder + "/berlin.gpkg")
# Restrict the reference_raster to the bbox_simulation
# check the CRS of bbox_simulation and reference_raster
if bbox_subset.crs != reference_raster.rio.crs:
    print("CRS of bbox_subset and reference_raster do not match. Reprojecting bbox_subset.")
    bbox_subset = bbox_subset.to_crs(reference_raster.rio.crs)
reference_raster = reference_raster.rio.clip(bbox_subset.geometry, bbox_subset.crs, drop=True)
reference_raster.plot()    


# retrieve the transform and CRS from the reference_raster
ref_transform = reference_raster.rio.transform()
ref_crs = reference_raster.rio.crs
# print the CRS and transform to check
print("CRS of the reference raster:", ref_crs)
print("Transform of the reference raster:", ref_transform)

# load the datasets to rasterize
railways = gpd.read_file(datafolder + "/city/osm_railways_without_subway_bbox.gpkg")
subway = gpd.read_file(datafolder + "/city/osm_subway_berlin.gpkg")
roads = gpd.read_file(datafolder + "/city/osm_major_roads_bbox.gpkg")
water = gpd.read_file(datafolder + "/city/osm_water_bbox.gpkg")
greens = gpd.read_file(datafolder + "/city/osm_green_without_forest_bbox.gpkg")
forest = gpd.read_file(datafolder + "/city/osm_forest_bbox.gpkg")
buildings = gpd.read_file(datafolder + "/city/osm_buildings_bbox.gpkg")

# load the datasets to rasterize - subset for prediction
railways = gpd.read_file(datafolder + "/prediction-subset/railways-scenario.gpkg")
subway = gpd.read_file(datafolder + "/prediction-subset/subway-scenario.gpkg")
roads = gpd.read_file(datafolder + "/prediction-subset/roads-scenario.gpkg")
water = gpd.read_file(datafolder + "/prediction-subset/water-scenario.gpkg")
greens = gpd.read_file(datafolder + "/prediction-subset/greens-scenario.gpkg")
forest = gpd.read_file(datafolder + "/prediction-subset/forest-scenario.gpkg")
buildings = gpd.read_file(datafolder + "/prediction-subset/buildings-scenario.gpkg")


# loop through the datasets and rasterize them on the same grid as the reference_raster
datasets = {
    "railways": railways,
    "subway": subway,
    "roads": roads,
    "water": water,
    "greens": greens,
    "forest": forest,
    "buildings": buildings
}
for name, dataset in datasets.items():
    # rasterize the dataset
    rasterized = rasterize(
        [(geom, 1) for geom in dataset.geometry],
        out_shape=reference_raster.shape[1:],
        transform=ref_transform,
        fill=0,
        all_touched=True
    )
    
    # write the rasterized dataset to a GeoTIFF file
    output_filename = os.path.join(datafolder, f"{name}_raster.tif")
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=rasterized.shape[0],
        width=rasterized.shape[1],
        count=1,
        dtype=rasterized.dtype,
        crs=ref_crs,
        transform=ref_transform
    ) as dst:
        dst.write(rasterized, 1)
    print(f"Rasterized {name} and saved to {output_filename}")

