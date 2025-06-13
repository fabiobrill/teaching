import os
import numpy as np
import geopandas as gpd
import rioxarray
import rasterio

datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data"

# load the file "/uhi/T2M_daily_mean_max_topography_2011_2020_present_30.tif" as heat
heat_raster = rioxarray.open_rasterio(os.path.join(datafolder, "uhi/T2M_daily_mean_max_topography_2011_2020_present_30.tif")).squeeze()
heat_raster.plot()

# load the convolved feature rasters
greens_raster = rioxarray.open_rasterio(os.path.join(datafolder, "greens_convolved_990m.tif")).squeeze()
forest_raster = rioxarray.open_rasterio(os.path.join(datafolder, "forest_convolved_990m.tif")).squeeze()
water_raster = rioxarray.open_rasterio(os.path.join(datafolder, "water_convolved_990m.tif")).squeeze()
railways_raster = rioxarray.open_rasterio(os.path.join(datafolder, "railways_convolved_990m.tif")).squeeze()
subway_raster = rioxarray.open_rasterio(os.path.join(datafolder, "subway_convolved_990m.tif")).squeeze()
roads_raster = rioxarray.open_rasterio(os.path.join(datafolder, "roads_convolved_990m.tif")).squeeze()
buildings_raster = rioxarray.open_rasterio(os.path.join(datafolder, "buildings_convolved_990m.tif")).squeeze()

# stack them all, with heat as the first layer
stacked_raster = np.stack([heat_raster, greens_raster, forest_raster,
                           water_raster, railways_raster, subway_raster,
                           roads_raster, buildings_raster], axis=0)
stacked_raster.shape # should be (8, height, width) where 8 is the number of layers

# slice the stacked raster into 4 tiles
def slice_into_tiles(raster, tile_size):
    """Slice a raster into tiles of specified size."""
    height, width = raster.shape[1], raster.shape[2]
    tiles = []
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = raster[:, i:i + tile_size, j:j + tile_size]
            if tile.shape[1] == tile_size and tile.shape[2] == tile_size:
                tiles.append(tile)
    return np.array(tiles)

tile_size = stacked_raster.shape[1] // 2  # Assuming we want 2x2 tiles
tiles = slice_into_tiles(stacked_raster, tile_size)
tiles.shape

# the geolocation of the tiles is based on the original raster's transform and CRS
# but we need to adjust the transform for each tile
def get_tile_transform(original_transform, tile_index, tile_size):
    """Calculate the transform for a specific tile based on the original transform."""
    row, col = tile_index
    new_transform = original_transform * rasterio.Affine.translation(col * tile_size, row * tile_size)
    return new_transform
def save_tiles_as_geotiff(tiles, original_transform, original_crs, output_folder):
    """Save the tiles as individual GeoTIFF files with correct georeferencing."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, tile in enumerate(tiles):
        tile_index = (idx // (tiles.shape[0] // 2), idx % (tiles.shape[0] // 2))  # Calculate tile index
        tile_transform = get_tile_transform(original_transform, tile_index, tile.shape[1])
        
        output_filename = os.path.join(output_folder, f"tile_{idx}.tif")
        with rasterio.open(
            output_filename,
            'w',
            driver='GTiff',
            height=tile.shape[1],
            width=tile.shape[2],
            count=tile.shape[0],
            dtype=tile.dtype,
            crs=original_crs,
            transform=tile_transform
        ) as dst:
            dst.write(tile)
        print(f"Saved tile {idx} to {output_filename}")
output_folder = os.path.join(datafolder, "tiles")
save_tiles_as_geotiff(tiles, heat_raster.rio.transform(), heat_raster.rio.crs, output_folder)