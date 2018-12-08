import os
from PIL import Image
import shutil

data_path = "./data/256_ObjectCategories/train"

dirs = os.listdir(data_path)
dirs.sort()

with open(r'label.txt', 'w', encoding='utf-8') as f:
	for i in dirs:
		f.write(i)
		f.write('\n')

it = 0
Matrix = [[] for x in range(257)]
for d in dirs:
	for _, _, filename in os.walk(os.path.join(data_path, d)):
		for i in filename:
			Matrix[it].append(os.path.join(os.path.join(data_path, d), i))
	it = it + 1


for i in range(len(Matrix)):
	for j in range(10):
		label = Matrix[i][j].split('/')[4]
		path = './data/256_ObjectCategories/test/' + label
		if not os.path.exists(path):
			os.mkdir(path)
		fullpath = path + '/' + Matrix[i][j].split('/')[5]
		shutil.move(Matrix[i][j], fullpath)


'''
with open(r'dataset-test.txt', 'w', encoding='utf-8') as f:
	for i in range(len(Matrix)):
		for j in range(10):
			f.write(Matrix[i][j])
			# f.write(os.path.join(data_path, Matrix[i][j]))
			f.write(' ')
			f.write(str(i))
			f.write('\n')

with open(r'dataset-train.txt', 'w', encoding='utf-8') as f:
	for i in range(len(Matrix)):
		for j in range(10, len(Matrix[i])):
			f.write(Matrix[i][j])
			# f.write(os.path.join(data_path, Matrix[i][j]))
			f.write(' ')
			f.write(str(i))
			f.write('\n')
'''
