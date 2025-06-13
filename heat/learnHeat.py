


datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/tiles"

# write a script that reads tile_0.tif and randomly selects 10% of the pixels
# then uses scikit-learn to train a linear regression model
# where y is the first layer (heat) and X is all other layers 
# (greens, forest, water, railways, subway, roads, buildings)

import os
import rasterio
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import geopandas as gpd
from sklearn.model_selection import train_test_split
# Load the raster data
tile_path = os.path.join(datafolder, "tile_0.tif")
with rasterio.open(tile_path) as src:
    tile_data = src.read()  # Read all layers
    transform = src.transform
    crs = src.crs
    height, width = tile_data.shape[1], tile_data.shape[2]

# Reshape the data for modeling
tile_data_reshaped = tile_data.reshape(tile_data.shape[0], -1).T  # Shape: (num_pixels, num_layers)
# Separate the heat layer (first layer) and the features (remaining layers)
heat_layer = tile_data_reshaped[:, 0]  # First layer is heat
features = tile_data_reshaped[:, 1:]  # Remaining layers are features
# Randomly select 10% of the pixels
num_pixels = features.shape[0]
sample_size = int(num_pixels * 0.1)
random_indices = random.sample(range(num_pixels), sample_size)
# Create the sample for training
X_sample = features[random_indices]
y_sample = heat_layer[random_indices]
# Split the sample into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)


# train a random forest regressor model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.4f}")
# Predict the heat layer using the model
predicted_heat = model.predict(features)
# Reshape the predicted heat back to the original tile shape
predicted_heat_reshaped = predicted_heat.reshape(height, width)
# Save the predicted heat layer as a new GeoTIFF file
output_path = os.path.join(datafolder, "predicted_heat_tile_0_rf.tif")
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=predicted_heat_reshaped.shape[0],
    width=predicted_heat_reshaped.shape[1],
    count=1,
    dtype=predicted_heat_reshaped.dtype,
    crs=crs,
    transform=transform
) as dst:
    dst.write(predicted_heat_reshaped, 1)
print(f"Predicted heat layer saved to {output_path}")
# Load the original tile to visualize the predicted heat layer
original_tile = rasterio.open(tile_path).read(1)  # Read the first layer (heat) of the original tile
# Visualize the predicted heat layer and the original heat layer, with identical dimensions
# make sure the colorscales of both images are identical
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Heat Layer")
plt.imshow(original_tile, cmap='hot', vmin=np.min(original_tile), vmax=np.max(original_tile), interpolation='nearest')
plt.colorbar(label='Heat Value')
plt.subplot(1, 2, 2)
plt.title("Predicted Heat Layer")
plt.imshow(predicted_heat_reshaped, cmap='hot', vmin=np.min(original_tile), vmax=np.max(original_tile), interpolation='nearest')
plt.colorbar(label='Heat Value')
plt.tight_layout()
plt.show()

# visualize the difference between the original heat layer and the predicted heat layer in percentage
difference = (predicted_heat_reshaped - original_tile) / original_tile * 100  # Percentage difference
plt.figure(figsize=(8, 6))
plt.title("Percentage Difference between Original and Predicted Heat Layer")
plt.imshow(difference, cmap='coolwarm', vmin=-100, vmax=100, interpolation='nearest')
plt.colorbar(label='Percentage Difference (%)')
plt.tight_layout()

# visualize the difference between the original heat layer and the predicted heat layer in absolute values
plt.figure(figsize=(8, 6))
plt.title("Absolute Difference between Original and Predicted Heat Layer")
plt.imshow(np.abs(predicted_heat_reshaped - original_tile), cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Absolute Difference')
plt.tight_layout()

# Visualize the coefficients of the random forest model
import matplotlib.pyplot as plt
# Note: Random Forest does not have coefficients like linear regression, but we can visualize feature importances
feature_importances = model.feature_importances_
# Create a bar plot of the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), ['greens', 'forest', 'water', 'railways', 'subway', 'roads', 'buildings'], rotation=45)
plt.title("Feature Importances of Random Forest Model")
plt.xlabel("Features")
plt.ylabel("Importance Value")
plt.grid(axis='y')
plt.tight_layout()

# plot the partial dependence plots for each feature
from sklearn.inspection import PartialDependenceDisplay
# Create a figure for partial dependence plots
fig, ax = plt.subplots(figsize=(12, 8))
# Plot partial dependence for each feature
features_names = ['greens', 'forest', 'water', 'railways', 'subway', 'roads', 'buildings']
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=range(len(features_names)),
    feature_names=features_names,
    ax=ax,
    grid_resolution=50
)



# visualize the coefficients of the linear regression model
coefficients = model.coef_
# Create a bar plot of the coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xticks(range(len(coefficients)), ['greens', 'forest', 'water', 'railways', 'subway', 'roads', 'buildings'], rotation=45)
plt.title("Coefficients of Linear Regression Model")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# now load tile_1.tif and predict the heat layer using the same model
tile_path_1 = os.path.join(datafolder, "tile_1.tif")
with rasterio.open(tile_path_1) as src:
    tile_data_1 = src.read()  # Read all layers
    transform_1 = src.transform
    crs_1 = src.crs
    height_1, width_1 = tile_data_1.shape[1], tile_data_1.shape[2]
# Reshape the data for modeling
tile_data_reshaped_1 = tile_data_1.reshape(tile_data_1.shape[0], -1).T  # Shape: (num_pixels, num_layers)
# Separate the heat layer (first layer) and the features (remaining layers)
heat_layer_1 = tile_data_reshaped_1[:, 0]  # First layer is heat
features_1 = tile_data_reshaped_1[:, 1:]  # Remaining layers are features
# Predict the heat layer using the model
predicted_heat_1 = model.predict(features_1)
# Reshape the predicted heat back to the original tile shape
predicted_heat_reshaped_1 = predicted_heat_1.reshape(height_1, width_1)
# Save the predicted heat layer as a new GeoTIFF file
output_path_1 = os.path.join(datafolder, "predicted_heat_tile_1_rf.tif")
with rasterio.open(
    output_path_1,
    'w',
    driver='GTiff',
    height=predicted_heat_reshaped_1.shape[0],
    width=predicted_heat_reshaped_1.shape[1],
    count=1,
    dtype=predicted_heat_reshaped_1.dtype,
    crs=crs_1,
    transform=transform_1
) as dst:
    dst.write(predicted_heat_reshaped_1, 1)
print(f"Predicted heat layer for tile 1 saved to {output_path_1}")
# Visualize the predicted heat layer for tile 1
import matplotlib.pyplot as plt
# Load the original tile to visualize the predicted heat layer
original_tile_1 = rasterio.open(tile_path_1).read(1)  # Read the first layer (heat) of the original tile
# Visualize the predicted heat layer and the original heat layer, with identical dimensions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Heat Layer Tile 1")
plt.imshow(original_tile_1, cmap='hot', vmin=np.min(original_tile_1), vmax=np.max(original_tile_1), interpolation='nearest')
plt.colorbar(label='Heat Value')
plt.subplot(1, 2, 2)
plt.title("Predicted Heat Layer Tile 1")
plt.imshow(predicted_heat_reshaped_1, cmap='hot', vmin=np.min(original_tile_1), vmax=np.max(original_tile_1), interpolation='nearest')
plt.colorbar(label='Heat Value')
plt.tight_layout()
plt.show()
# Visualize the difference between the original heat layer and the predicted heat layer in percentage
difference_1 = (predicted_heat_reshaped_1 - original_tile_1) / original_tile_1 * 100  # Percentage difference
plt.figure(figsize=(8, 6))
plt.title("Percentage Difference between Original and Predicted Heat Layer Tile 1")
plt.imshow(difference_1, cmap='coolwarm', vmin=-100, vmax=100, interpolation='nearest')
plt.colorbar(label='Percentage Difference (%)')
plt.tight_layout()

# Visualize the difference between the original heat layer and the predicted heat layer in absolute values
plt.figure(figsize=(8, 6))
plt.title("Absolute Difference between Original and Predicted Heat Layer Tile 1")
plt.imshow(np.abs(predicted_heat_reshaped_1 - original_tile_1), cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Absolute Difference')
plt.tight_layout()

# evaluate the model on tile 1
from sklearn.metrics import mean_squared_error, r2_score
# Calculate the mean squared error and R^2 score
mse = mean_squared_error(original_tile_1.flatten(), predicted_heat_reshaped_1.flatten())
r2 = r2_score(original_tile_1.flatten(), predicted_heat_reshaped_1.flatten())
print(f"Mean Squared Error for Tile 1: {mse:.4f}")
print(f"R^2 Score for Tile 1: {r2:.4f}")


# write a function that takes a tile path and a model, and predicts the heat layer
def predict_heat_layer(tile_path, model, output_path):
    with rasterio.open(tile_path) as src:
        tile_data = src.read()  # Read all layers
        transform = src.transform
        crs = src.crs
        height, width = tile_data.shape[1], tile_data.shape[2]
    
    # Reshape the data for modeling
    tile_data_reshaped = tile_data.reshape(tile_data.shape[0], -1).T  # Shape: (num_pixels, num_layers)
    # Separate the heat layer (first layer) and the features (remaining layers)
    features = tile_data_reshaped[:, 1:]  # Remaining layers are features
    
    # Predict the heat layer using the model
    predicted_heat = model.predict(features)
    
    # Reshape the predicted heat back to the original tile shape
    predicted_heat_reshaped = predicted_heat.reshape(height, width)
    
    # Save the predicted heat layer as a new GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=predicted_heat_reshaped.shape[0],
        width=predicted_heat_reshaped.shape[1],
        count=1,
        dtype=predicted_heat_reshaped.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(predicted_heat_reshaped, 1)
    
    print(f"Predicted heat layer saved to {output_path}")
# Example usage of the function
tile_path_2 = os.path.join(datafolder, "tile_2.tif")
output_path_2 = os.path.join(datafolder, "predicted_heat_tile_2_rf.tif")
predict_heat_layer(tile_path_2, model, output_path_2)

tile_path_3 = os.path.join(datafolder, "tile_3.tif")
output_path_3 = os.path.join(datafolder, "predicted_heat_tile_3_rf.tif")
predict_heat_layer(tile_path_3, model, output_path_3)