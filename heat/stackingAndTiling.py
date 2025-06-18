import os
import numpy as np
import rioxarray
import rasterio

from customFunctions import slice_into_tiles, save_tiles_as_geotiff, save_stacked_raster_as_geotiff

# load the file "/uhi/T2M_daily_mean_max_topography_2011_2020_present_30.tif" as heat
datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data"
heat_raster = rioxarray.open_rasterio(os.path.join(datafolder, "uhi/T2M_daily_mean_max_topography_2011_2020_present_30.tif")).squeeze()
#datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/prediction-subset"
#heat_raster = rioxarray.open_rasterio(os.path.join(datafolder, "reference_raster_clipped.tif")).squeeze()
heat_raster.plot()

# load the convolved feature rasters
greens_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "greens_convolved_990m.tif")).squeeze()
forest_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "forest_convolved_990m.tif")).squeeze()
water_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "water_convolved_990m.tif")).squeeze()
railways_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "railways_convolved_990m.tif")).squeeze()
subway_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "subway_convolved_990m.tif")).squeeze()
roads_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "roads_convolved_990m.tif")).squeeze()
buildings_convolved= rioxarray.open_rasterio(os.path.join(datafolder, "buildings_convolved_990m.tif")).squeeze()

# load the distance feature rasters
greens_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_greens.tif")).squeeze()
forest_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_forest.tif")).squeeze()
water_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_water.tif")).squeeze()
railways_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_railways.tif")).squeeze()
subway_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_subway.tif")).squeeze()
roads_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_roads.tif")).squeeze()
buildings_distance = rioxarray.open_rasterio(os.path.join(datafolder, "distance_to_nearest_buildings.tif")).squeeze()

# stack them all, with heat as the first layer
stacked_raster = np.stack([heat_raster, greens_convolved, forest_convolved,
                           water_convolved, railways_convolved, subway_convolved,
                           roads_convolved, buildings_convolved,
                           greens_distance, forest_distance, water_distance,
                           railways_distance, subway_distance, roads_distance,
                           buildings_distance], axis=0)
stacked_raster.shape # should be (15, height, width) where 15 is the number of layers

# slice the stacked raster into 4 tiles, using the provided custom function
tile_size = stacked_raster.shape[1] // 2  # Assuming we want 2x2 tiles
tiles = slice_into_tiles(stacked_raster, tile_size)
tiles.shape

# the geolocation of the tiles is based on the original raster's transform and CRS
# but we need to adjust the transform for each tile
output_folder = os.path.join(datafolder, "tiles")
save_tiles_as_geotiff(tiles, heat_raster.rio.transform(), heat_raster.rio.crs, output_folder)

# alternative version: save full stacked raster as a single GeoTIFF
output_filename = os.path.join(datafolder, "stacked_raster.tif")
save_stacked_raster_as_geotiff(stacked_raster, heat_raster.rio.transform(), heat_raster.rio.crs, output_filename)