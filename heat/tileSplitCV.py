from sklearn.model_selection import BaseCrossValidator

# Custom cross-validator to split tiles
class TileSplitCV(BaseCrossValidator):
    def __init__(self, tile_0_indices, tile_3_indices):
        self.tile_0_indices = tile_0_indices
        self.tile_3_indices = tile_3_indices

    def split(self, X, y=None, groups=None):
        yield self.tile_0_indices, self.tile_3_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1  # Only one split for this custom cross-validation

# write a unit test for the TileSplitCV class
def test_tile_split_cv():
    # Create a small dataset for testing
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    
    # Create indices for tile_0 and tile_3
    tile_0_indices = np.array([0, 1])  # First two samples for tile_0
    tile_3_indices = np.array([2, 3])  # Last two samples for tile_3
    
    # Create the custom cross-validator
    cv = TileSplitCV(tile_0_indices, tile_3_indices)
    
    # Check the split method
    splits = list(cv.split(X, y))
    assert len(splits) == 1, "Should yield one split"
    train_indices, test_indices = splits[0]
    
    # Check if the training and testing indices are correct
    assert np.array_equal(train_indices, tile_0_indices), "Training indices should match tile_0"
    assert np.array_equal(test_indices, tile_3_indices), "Testing indices should match tile_3"
    # Check the number of splits
    assert cv.get_n_splits() == 1, "Should return one split"
    
# Run the unit test
test_tile_split_cv()


# use samples from tile_0 as training and from tile_3 as testing WITHIN the grid search
# tile_data_reshaped_0 contains the training data from tile_0
# tile_data_reshaped_3 contains the testing data from tile_3
# Define the training and testing data
X_train = tile_data_reshaped_0[:, 1:]  # Features from tile_0
y_train = tile_data_reshaped_0[:, 0]  # Heat layer from tile_0
X_test = tile_data_reshaped_3[:, 1:]  # Features from tile_3
y_test = tile_data_reshaped_3[:, 0]  # Heat layer from tile_3

# define the grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 7, 9, 15]
}
# Create a Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Set up a grid search that uses training and test data from different tiles
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
# Fit the grid search to the training data
grid_search.fit(X_train, y_train)
# that's wrong, we need to use the training data from tile_0 and the testing data from tile_3
# the GridSearchCV function from scikit-learn does not support using different datasets for training and testing directly.
# Instead, we can manually split the data and predict the test data,
# but we need to find a solution that allows us to optimize the hyperparameters based on the prediction skill on the test data.


# To achieve this, we can use the `cross_val_score` function with a custom cross-validation strategy
# that allows us to use the training data from tile_0 and the testing data from tile_3.
# However, this requires a custom cross-validation strategy that separates the training and testing data based on the tiles.
# we will code this now


# Create indices for tile_0 and tile_3
# remember that training_data contains the merged data from tile_0 and tile_3
# and that the indices for tile_0 and tile_3 are identical,
# but we need to separate them for the custom cross-validation
# however, within training_data, the first part corresponds to tile_0 and the second part to tile_3
tile_0_indices = np.arange(tile_data_reshaped_0.shape[0])  # Indices for tile_0
tile_3_indices = np.arange(tile_data_reshaped_3.shape[0]) + tile_data_reshaped_0.shape[0]  # Indices for tile_3





# randomly sample 10% of tile_data_0 and call it subdata_0
subdata_0 = tile_data_reshaped_0[np.random.choice(tile_data_reshaped_0.shape[0], int(tile_data_reshaped_0.shape[0] * 0.1), replace=False)]
subdata_3 = tile_data_reshaped_3[np.random.choice(tile_data_reshaped_3.shape[0], int(tile_data_reshaped_3.shape[0] * 0.1), replace=False)]

# define training data (in this case merging tiles 0 and 3)
subdata_training = np.vstack((subdata_0, subdata_3))
tile_0_indices = np.arange(subdata_0.shape[0])
tile_3_indices = np.arange(subdata_3.shape[0]) + subdata_0.shape[0]  # Indices for tile_3

tile_split_cv = TileSplitCV(tile_0_indices, tile_3_indices)

# Now we can use this custom cross-validator in the grid search
# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 7, 9, 15]
}
# Create a Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
# Set up a grid search that uses the custom cross-validation strategy
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tile_split_cv, scoring='r2', n_jobs=-1)
# Fit the grid search to the training data, using the custom cross-validation strategy
grid_search.fit(subdata_training[:, 1:], subdata_training[:, 0])  # Use the reshaped training data
# Note: The training data is the merged data from tile_0 and tile_3,
# and the heat layer is the first layer of the reshaped data.
# the splitting is done by the custom cross-validator, which separates the training and testing data based on the tiles.

# please think for a moment about the code above: is it doing what we want?
# Yes, the code above is correctly implementing a custom cross-validation strategy that separates the training data from tile_0 and the testing data from tile_3.
# How does it work?
# The `TileSplitCV` class inherits from `BaseCrossValidator` and defines a custom split method that yields the indices for training and testing data.
# The `split` method yields the indices for tile_0 and tile_3, allowing the grid search to use these indices for training and testing.


# print the ranking of estimators with hyperparameters and skill score as table
results = pd.DataFrame(grid_search.cv_results_)
# Display the results in a table format
print("Grid Search Results:")
print(results[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score'))





# Use the features from tile_0 and the heat layer from tile_0 for training
# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best parameters from grid search with tile split: {best_params}")
# Now we can use the best parameters to create a new Random Forest Regressor
# Create a Random Forest Regressor with the best parameters
rf_best_tile_split = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    n_jobs=-1,
    random_state=42
)
# Fit the model with the best parameters to the training data from tile_0
rf_best_tile_split.fit(tile_data_reshaped_0[:, 1:], tile_data_reshaped_0[:, 0])
# Now we can predict the heat layer for tile_3 using the trained model
predicted_heat_tile_3 = rf_best_tile_split.predict(tile_data_reshaped_3[:, 1:])

# evaluate the model on all 4 tiles
# Calculate the mean squared error and R^2 score for tile_3
from sklearn.metrics import mean_squared_error, r2_score
mse_tile_3 = mean_squared_error(tile_data_reshaped_3[:, 0], predicted_heat_tile_3)
r2_tile_3 = r2_score(tile_data_reshaped_3[:, 0], predicted_heat_tile_3)
print(f"Mean Squared Error for Tile 3: {mse_tile_3:.4f}")
print(f"R^2 Score for Tile 3: {r2_tile_3:.4f}")



# save model to a file
import joblib
model_path = os.path.join(datafolder, "random_forest_model.pkl")
joblib.dump(model, model_path)


