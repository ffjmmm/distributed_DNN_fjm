import os

data_path = "./data/101_ObjectCategories"

d = os.listdir(data_path)
d.sort()

with open(r'./data/label_101.txt', 'w', encoding='utf-8') as f:
	for i in d:
		f.write(i)
		f.write('\n')

