


#datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/tiles-nightover20-maximum"
datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/tiles"

# write a script that reads tile_0.tif and randomly selects 10% of the pixels
# then uses scikit-learn to train a linear regression model
# where y is the first layer (heat) and X is all other layers 
# (greens, forest, water, railways, subway, roads, buildings)

import os
import rasterio
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import custom functions from another Python script in the same folder
from customFunctions import readRaster, writeRaster, plotSideBySide, plotSpatialError


# -1- Reading raster files & sampling training data

# read data from tiles 0 and 3, reshape, and merge
#data_0, transform_0, crs_0, height_0, width_0 = readRaster(os.path.join(datafolder, "tile_0.tif"))
#data_3, transform_3, crs_3, height_3, width_3 = readRaster(os.path.join(datafolder, "tile_3.tif"))
#data_reshaped_0 = data_0.reshape(data_0.shape[0], -1).T  # (num_pixels, num_layers)
#data_reshaped_3 = data_3.reshape(data_3.shape[0], -1).T
#training_data = np.vstack((data_reshaped_0, data_reshaped_3)) # merge the two tiles
data, transform, crs, height, width = readRaster(os.path.join(datafolder, "tile_0.tif"))
training_data = data.reshape(data.shape[0], -1).T  # Reshape to (num_pixels, num_layers)

# Separate the heat layer (first layer) and the features (remaining layers)
heat_layer = training_data[:, 0]  # First layer is heat
features = training_data[:, 1:]   # Remaining layers are features

# Randomly select a fraction of the pixels for training, e.g. 0.1 for 10%
fraction = 0.10
num_pixels = features.shape[0]
sample_size = int(num_pixels * fraction)
random_indices = random.sample(range(num_pixels), sample_size)
X_sample = features[random_indices]    # the predictors (features)
y_sample = heat_layer[random_indices]  # the target variable (heat layer)

# Split the sample into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)


# -2- Hyperparameter tuning with grid search

# implement a gridsearch with cross-validation to find the best hyperparameters for the random forest regressor
# Define the parameter grid for Random Forest
param_grid = {'n_estimators': [10, 20, 30, 40, 50],
              'max_depth': [None],
              'n_jobs': [-1]}  # Use all available cores
# Create a Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
# Fit the grid search to the training data
grid_search.fit(X_train, y_train)
# print the ranking of estimators with hyperparameters and skill score as table
results = pd.DataFrame(grid_search.cv_results_)
print(results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score'))

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best parameters from grid search: {best_params}")


# -3- Model fitting and evaluation

# train a model with the best parameters (manually set for simplicity)
model = RandomForestRegressor(n_estimators=10, max_depth=None, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.4f}")
# predict the heat layer using the model
predicted_heat = model.predict(features)
# reshape the predicted heat back to the original tile shape
predicted_heat_reshaped = predicted_heat.reshape(height, width)
# use the custom function to plot the original and predicted heat layers side by side
plotSideBySide(data[0], predicted_heat_reshaped)
plotSpatialError(data[0], predicted_heat_reshaped)

# save the predicted heat layer as a new GeoTIFF file
output_path = os.path.join(datafolder, "predicted_heat_tile_0_rf.tif")
writeRaster(predicted_heat_reshaped, output_path, transform, crs)
# save the fitted model to a file
model_path = os.path.join(datafolder, "random_forest_model.pkl")
joblib.dump(model, model_path)


# -4- Evaluate on tile 1 (not used in training)

# now load tile_1.tif and predict the heat layer using the same model
tile_path_1 = os.path.join(datafolder, "tile_1.tif")
tile_1, transform_1, crs_1, height_1, width_1 = readRaster(tile_path_1)
tile_data_reshaped_1 = tile_1.reshape(tile_1.shape[0], -1).T  # (num_pixels, num_layers)
predicted_heat_tile_1 = model.predict(tile_data_reshaped_1[:, 1:])  # Use features only
predicted_heat_reshaped_1 = predicted_heat_tile_1.reshape(height_1, width_1)
# evaluate the model on tile 1
mse = mean_squared_error(original_tile_1.flatten(), predicted_heat_reshaped_1.flatten())
r2 = r2_score(original_tile_1.flatten(), predicted_heat_reshaped_1.flatten())
print(f"Mean Squared Error for Tile 1: {mse:.4f}")
print(f"R^2 Score for Tile 1: {r2:.4f}")
# plot the original and predicted heat layers side by side
plotSideBySide(tile_1[0], predicted_heat_reshaped_1, title1="Original Heat Layer Tile 1", title2="Predicted Heat Layer Tile 1")
plotSpatialError(tile_1[0], predicted_heat_reshaped_1)


# -5- Model inspection

# caution: this is hard-coded for the specific features used in the model
# Define the feature names based on the layers in the raster data
features_names = ['greens_convolved', 'forest_convolved', 'water_convolved',
                  'railways_convolved', 'subway_convolved', 'roads_convolved',
                  'buildings_convolved', 'greens_distance', 'forest_distance',
                  'water_distance', 'railways_distance', 'subway_distance',
                  'roads_distance', 'buildings_distance']

# Visualize the feature importances
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), features_names, rotation=45)
plt.title("Feature Importances of Random Forest Model")
plt.xlabel("Features")
plt.ylabel("Importance Value")
plt.grid(axis='y')
plt.tight_layout()

# plot the partial dependence plots for each feature
fig, ax = plt.subplots(figsize=(12, 18))
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=range(len(features_names)),
    feature_names=features_names,
    ax=ax,
    #grid_resolution=200
)