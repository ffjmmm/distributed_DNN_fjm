# Distribued Neural Network on Edge Devices

1. Put **256_ObjectCategories.tar** in data and unzip it.

2. Caltech01.py and Caltech02.py are designed to preprocess the dataset, use ```python3 Caltech01.py``` to generate *label.txt* and use ```python3 Caltech02.py``` to generate *dataset-train.txt* and *dataset-test.txt*.

3. Use ```python3 main.py``` to run. Use ```python3 main.py --help``` to see some parameters.

4. New way to mimic the edge pixels loss. But it really costs **Too** much time.

    * Use markov_rand.
    
    ![](img/markov_rand.png)
    
    * Use dropout.
    
    ![](img/dropout.png)
    
5. Lossy Linear also spends much more time than original linear.

	* Lossy Linear.
	
	![](img/lossy_linear.png)
	
	* Original Linear.

	![](img/no_lossy_linear.png)
	