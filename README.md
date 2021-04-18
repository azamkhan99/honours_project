Code used for Sensing Spaces Project.

Steps to pre-process Airspeck Data:

1. Download PEEPs, Leon and Guadalajara Data. (https://drive.google.com/drive/folders/1CjZxHvWGVrYkjk0T7U5f7EYmT63o-T-4?usp=sharing)
2. Change paths to raw airspeck files in constants.py.
3. Execute cells in preprocessing_x_notebook.ipynb to preprocess and save raw files.
4. Execute cells in mexico-cleaning.ipynb to remove outliers from Leon and Guadalajara datasets.
5. Run cells in osm-data.ipynb to add road type data (Delhi.geojson should be downloaded separately).
