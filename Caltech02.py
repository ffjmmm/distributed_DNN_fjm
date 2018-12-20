import os
from PIL import Image
import shutil
import math

data_path = "./data/101_ObjectCategories"

dirs = os.listdir(data_path)
dirs.sort()

with open(r'./data/label_101.txt', 'w', encoding='utf-8') as f:
	for i in dirs:
		f.write(i)
		f.write('\n')

it = 0
Matrix = [[] for x in range(101)]
for d in dirs:
	for _, _, filename in os.walk(os.path.join(data_path, d)):
		for i in filename:
			Matrix[it].append(os.path.join(os.path.join(data_path, d), i))
	it = it + 1

with open(r'./data/dataset-test-101.txt', 'w', encoding='utf-8') as f:
    for i in range(len(Matrix)):
        num_test = math.floor(len(Matrix[i]) * 0.3)
        for j in range(num_test):
            f.write(Matrix[i][j])
			# f.write(os.path.join(data_path, Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')

with open(r'./data/dataset-train-101.txt', 'w', encoding='utf-8') as f:
    for i in range(len(Matrix)):
        num_test = math.floor(len(Matrix[i]) * 0.3)
        for j in range(num_test, len(Matrix[i])):
            f.write(Matrix[i][j])
			# f.write(os.path.join(data_path, Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
