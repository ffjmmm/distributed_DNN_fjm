# Distribued Neural Network on Edge Devices

1. Put **256_ObjectCategories.tar** in data and unzip it.
2. Caltech01.py and Caltech02.py are designed to preprocess the dataset, use ```python3 Caltech02.py``` to get *dataset-test.txt* and *data-train.txt*, which are used to load the data.
3. Use ```python3 main.py --dataset 'Caltech'``` to run.
4. I will change the code tomorrow to plot the acc curve and loss curve, get the time that each layer spends, and compare the distributed model with original model.
