Code used for Sensing Spaces Project.

Steps to pre-process Airspeck Data:

1. Download PEEPs, Leon and Guadalajara Data. (datasync)
2. Download Delhi.geojson and place in preprocessing folder.
3. Change paths to raw airspeck files in constants.py.
4. Execute cells in preprocessing_x_notebook.ipynb to preprocess and save raw files.
5. Execute cells in mexico-cleaning.ipynb to remove outliers from Leon and Guadalajara datasets.
6. Run cells in osm-data.ipynb to add road type data (Delhi.geojson should be downloaded separately).

- eda.ipynb contains cells which compute the standard deviation and mean of the PM2.5 levels in each dataset (Airspeck-P and Airspeck-S).

This project is divided into four components: Spatial, Temporal, Spatiotemporal, and Extensibility.

Spatial:
- baseline.ipynb allows running various machine-learning methods on the datasets to produce spatial interpolations. The notebook provides the MAPE, MAE, and RMSE of each method for each dataset.
- spatial-T.ipynb includes cells which run extra-trees on the spatial-T datasets with various window-sizes, and report the error metrics.

Temporal:

- temporal-preds.ipynb is a notebook for running experiments and testing the architecture of GRU and LSTM models. It also contains code for visualising predictions.
- gru_hyperparameter_tuning.ipynb and lstm-hyperparameter-tuning.ipynb both include code for running an exhaustive grid search to find the best-performing temporal models. TensorBoard is used to see the results in terms of MAPE.
- fut-ts-plots.ipynb is a notebook which contains very simple code that produces plots showing the increase in MAPE as the prediction time is increased. The MAPE is computed using temporal-preds.ipynb with optimal hyperparameters and the results are saved in an excel file.

Spatiotemporal:

- approachB_making_dataset.ipynb includes code for producing spatial interpolations at subsequent timesteps for each Airspeck-P datapoint. 
- approachB_tuning.ipynb is used for performing a grid search to find the best hyperparameters of the temporal model.

Extensibility:

- generalisation\ copy.ipynb contains code for performing the extensibility and generalisation experiments, and viewing the results. The datasets are first loaded, and Extra-Trees models are trained on each. Each model is then applied to every dataset and the MAPE is computed.
