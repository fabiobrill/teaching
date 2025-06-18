import os
import rasterio
import numpy as np

def readRaster(input_path):
    """Read a GeoTIFF file and return the data, transform, and CRS."""
    with rasterio.open(input_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        height, width = data.shape[1], data.shape[2]
    return data, transform, crs, height, width

def writeRaster(data, output_path, transform, crs):
    """Write a numpy array to a GeoTIFF file."""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data, 1)
    print(f"Saved file {output_path}")

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

def save_stacked_raster_as_geotiff(stacked_raster, original_transform, original_crs, output_filename):
    """Save the stacked raster as a single GeoTIFF file with correct georeferencing."""
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=stacked_raster.shape[1],
        width=stacked_raster.shape[2],
        count=stacked_raster.shape[0],
        dtype=stacked_raster.dtype,
        crs=original_crs,
        transform=original_transform
    ) as dst:
        dst.write(stacked_raster)
    print(f"Saved stacked raster to {output_filename}")

# write a function that takes a tile path and a model, and predicts the heat layer
def predictHeatLayer(input_path, model, output_path, features_only=False):
    """
    Predict the heat layer from a GeoTIFF tile using a trained model.
    This function assumes the first layer is the target (heat) layer and the remaining layers are features.
    If a raster stack without a heat layer is provided, set features_only=True.
    """
    # read the input raster tile
    input_data, transform, crs, height, width = readRaster(input_path)
    # Reshape the data for modeling
    input_data_reshaped = input_data.reshape(input_data.shape[0], -1).T
    # Separate the heat layer (first layer) and the features (remaining layers)
    if features_only:
        features = input_data_reshaped  # Use all layers as features
    else:
        features = input_data_reshaped[:, 1:]  # first layer is target, remaining layers are features
    # Predict the heat layer using the model
    predicted_heat = model.predict(features)
    # Reshape the predicted heat layer back to 2D and save as a new GeoTIFF file
    predicted_heat_reshaped = predicted_heat.reshape(height, width)
    writeRaster(predicted_heat_reshaped, output_path, transform, crs)

def plotSideBySide(original, predicted, title1="Original", title2="Predicted"):
    """Plot two images side by side, using the same color scale."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(original, cmap='hot', vmin=np.min(original), vmax=np.max(original), interpolation='nearest')
    plt.colorbar(label='Heat Value')
    
    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(predicted, cmap='hot', vmin=np.min(original), vmax=np.max(original), interpolation='nearest')
    plt.colorbar(label='Heat Value')
    
    plt.tight_layout()
    plt.show()

def plotSpatialError(original, predicted):
    """Plot the spatial error between original and predicted heat layers."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Absolute Difference")
    plt.imshow((predicted - original), cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Units of target variable')

    plt.subplot(1, 2, 2)
    plt.title("Percentage Difference")
    plt.imshow(((predicted - original) / original * 100), cmap='coolwarm', vmin=-100, vmax=100, interpolation='nearest')
    plt.colorbar(label='Percentage (%)')

    plt.tight_layout()
    plt.show()
