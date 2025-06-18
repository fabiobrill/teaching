import os
import joblib

# import custom functions from another Python script in the same folder
from customFunctions import readRaster, writeRaster, predictHeatLayer
help(predictHeatLayer) # check the docstring of the function

datafolder = "D:/Seafile/Meine Bibliothek/teaching/teaching-heat/data/prediction-subset"

rf_model = joblib.load(os.path.join(datafolder, "random_forest_model.pkl"))
predictHeatLayer(datafolder + "/stacked_raster.tif", rf_model, datafolder + "/prediction_subset.tif")

#predictHeatLayer(datafolder + "/tile_0.tif", rf_model, datafolder + "/prediction_test0.tif")
#predictHeatLayer(datafolder + "/tile_1.tif", rf_model, datafolder + "/prediction_test1.tif")
#predictHeatLayer(datafolder + "/tile_2.tif", rf_model, datafolder + "/prediction_test2.tif")
#predictHeatLayer(datafolder + "/tile_3.tif", rf_model, datafolder + "/prediction_test3.tif")