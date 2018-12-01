# Distribued Neural Network on Edge Devices

1. Put **256_ObjectCategories.tar** in data and unzip it.

2. Caltech01.py and Caltech02.py are designed to preprocess the dataset, use ```python3 Caltech02.py``` to get *dataset-test.txt* and *data-train.txt*, which are used to load the data.

3. Use ```python3 main.py``` to run. Use ```python3 main.py --help`` to see some parameters.

4. There are 5 features in the VGG model. We first divide the input *x* into 4 parts: *x11, x12, x21, x22*, and then apply the feature 1 and feature 2 on *x11, x12, x21, x22*. Then, in the feature 3 and feature 4, we use lossy conv2d to mimic the loss of the edge pixels. Third, we put the small parts together, and apply the feature 5. Finally, we flatten the matrix and use a FC layer to do the classify. Here, I compare the time that each feature spends.
	
	+ Use lossy_conv2d _new, batch size is 64.
	![](results/layer_time/new_lossy_conv.png)
	
	+ Do NOT use lossy_conv2d, batch size is 64.
	![](results/layer_time/no_lossy_conv.png)
	
	+ Use lossy_conv2d, batch size is 64.
	
		OOM, because too many matrixes are defined.
		
	+ Use lossy_conv2d, batch size is 32.
	![](results/layer_time/old_lossy_conv_32.png)
	
	+ Use lossy_conv2d _new, batch size is 32.
	![](results/layer_time/new_lossy_conv_32.png)
	
	
	Two mainly results:
	
	1. Using lossy_conv2d spends about 3 times more time than normal conv2d, but **I THINK** it's no so bad.
	
	2. The **new** lossy_conv2d only takes half the time of the **old** lossy_conv2d, and significantly reduces memory usage.
	
	
