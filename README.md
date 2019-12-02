# Synthetic City Generation

This includes the scripts necessary for both the data generation and the training and testing of models.

### data_generation
These scripts primarily involve the creation of synthetic data by interfacing with CityEngine.

1. **dynamic_shoot_syn_1_colorful_city.py** is run from within CityEngine twice to create the raw synthetic data patches: once with the city fully rendered for the image data, and a second time with buildings rendered as black and other extraneous objects taken out for the label file.

2. **pre_process_syn.py** reads in a directory filled with the raw tiles and formats the labels for use in the training processes. It also creates a file *colTileNames.txt* so the model can read in the list of files.

3. **randomizer.py** is used if only a subset of patches generated are needed or if you need to remake *colTileNames.txt*.

4. **crop.py** is used on real data from Inria as those tiles are 5000x5000 compared to DeepGlobe and synthetic tiles that are 572x572.


### experiments

1. **train_custom_threesources.py** pulls data from 3 separate sources in a 2:2:1 ratio for each mini batch and trains a model.

2. **test_custom_all.py** tests the above trained model on real images.

3. **iou.py** calculates either a pure average or weighted IoU for a given output result file from testing.

4. **crop_single.py** crops a single image based on given corner to start cropping at and desired subsection. Useful for zooming in on maps created by *visualize_predictions.py*

5. **visualize_predictions.py** allows the user to automatically compare model predictions to ground truths and then map the difference in color. Different types of errors are displayed in different colors, so you can visually compare performance.
