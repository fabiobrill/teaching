import os
import geopandas as gpd
import rioxarray
import rasterio
from rasterio.features import rasterize
import numpy as np
from scipy.ndimage import convolve

os.getcwd()

datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data"
os.listdir(datafolder)
# compute a 2d convolution with a 3x3 kernel for the greens raster

# Load the rasterized datasets, and ensure they are in float32 format
greens_raster = rioxarray.open_rasterio(os.path.join(datafolder, "greens_raster.tif")).squeeze().astype(np.float32)
forest_raster = rioxarray.open_rasterio(os.path.join(datafolder, "forest_raster.tif")).squeeze().astype(np.float32)
water_raster = rioxarray.open_rasterio(os.path.join(datafolder, "water_raster.tif")).squeeze().astype(np.float32)
railways_raster = rioxarray.open_rasterio(os.path.join(datafolder, "railways_raster.tif")).squeeze().astype(np.float32)
subway_raster = rioxarray.open_rasterio(os.path.join(datafolder, "subway_raster.tif")).squeeze().astype(np.float32)
roads_raster = rioxarray.open_rasterio(os.path.join(datafolder, "roads_raster.tif")).squeeze().astype(np.float32)
buildings_raster = rioxarray.open_rasterio(os.path.join(datafolder, "buildings_raster.tif")).squeeze().astype(np.float32)

# as 1 gridcell of the rasters is 30m, about 1000m is 33 gridcells (990m)
# therefore we define a kernel with 33x33 gridcells
kernel_size = 33
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)


# example
# Convolve the greens raster with the kernel
#convolved_greens = convolve(greens_raster, kernel, mode='constant', cval=0.0)

# loop through the datasets, compute the convolution and save them
datasets = {
    "greens": greens_raster,
    "forest": forest_raster,
    "water": water_raster,
    "railways": railways_raster,
    "subway": subway_raster,
    "roads": roads_raster,
    "buildings": buildings_raster
}
# compute the convolution for each dataset
convolved_datasets = {}
for name, raster in datasets.items():
    # Perform convolution with the defined kernel
    convolved_raster = convolve(raster, kernel, mode='constant', cval=0.0)
    convolved_datasets[name] = convolved_raster
    # Save the convolved raster to a new GeoTIFF file
    output_filename = os.path.join(datafolder, f"{name}_convolved_990m.tif")
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=convolved_raster.shape[0],
        width=convolved_raster.shape[1],
        count=1,
        dtype=convolved_raster.dtype,
        crs=raster.rio.crs,
        transform=raster.rio.transform()
    ) as dst:
        dst.write(convolved_raster, 1)


# compute distance to nearest railway, subway, road, water, greens, forest and buildings
from scipy.ndimage import distance_transform_edt
# Define a function to compute distance to nearest feature
def compute_distance_to_nearest_feature(raster, feature_name):
    # Invert the raster to get the distance to the nearest feature
    inverted_raster = np.where(raster > 0, 0, 1)
    # Compute the distance transform
    distance = distance_transform_edt(inverted_raster)
    # multiply by the grid size (30m) to convert to meters
    distance *= 30  # Assuming each grid cell is 30m x 30m
    # Save the distance raster to a new GeoTIFF file
    output_distance_filename = os.path.join(datafolder, f"distance_to_nearest_{feature_name}.tif")
    with rasterio.open(
        output_distance_filename,
        'w',
        driver='GTiff',
        height=distance.shape[0],
        width=distance.shape[1],
        count=1,
        dtype=distance.dtype,
        crs=raster.rio.crs,
        transform=raster.rio.transform()
    ) as dst:
        dst.write(distance, 1)

# Loop through the datasets and compute distance to nearest feature
for name, raster in datasets.items():
    compute_distance_to_nearest_feature(raster, name)



# Define Sobel kernels for x and y directions
sobel_x = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]], dtype=np.float32)
sobel_y = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]], dtype=np.float32)

# Perform convolution with Sobel kernels
sobel_x_result = convolve(forest_raster, sobel_x, mode='constant', cval=0.0)
sobel_y_result = convolve(forest_raster, sobel_y, mode='constant', cval=0.0)    
# Combine the results to get the magnitude of the gradient
sobel_magnitude = np.sqrt(sobel_x_result**2 + sobel_y_result**2)
# Save the Sobel magnitude result to a new GeoTIFF file
output_sobel_magnitude_filename = os.path.join(datafolder, "forest_sobel_magnitude.tif")
with rasterio.open(
    output_sobel_magnitude_filename,
    'w',
    driver='GTiff',
    height=sobel_magnitude.shape[0],
    width=sobel_magnitude.shape[1],
    count=1,
    dtype=sobel_magnitude.dtype,
    crs=ref_crs,
    transform=ref_transform
) as dst:
    dst.write(sobel_magnitude, 1)
