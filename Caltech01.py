import os

data_path = "./data/256_ObjectCategories/train"

d = os.listdir(data_path)
d.sort()

with open(r'label.txt', 'w', encoding='utf-8') as f:
	for i in d:
		f.write(i)
		f.write('\n')

