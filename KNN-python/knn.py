# an implementation for KNN classifier
# I just need to calculate the distance between the test data and all of the training data

from read_map_original import *
import datetime
import math
import random
import struct
import numpy as np
import scipy
import heapq

start_time = datetime.datetime.now()
train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()
num_images = 60000
num_test = 10000

 
log_file = 'log_knn'
log = open(log_file, 'w')

class Knn_classifier:
	def __init__(self, k):
		self.k = k

	def train(self, train_data, train_label):
		self.train_data = train_data
		self.train_label = train_label
		self.n_train = len(train_data)

	def predict(self, test_data):
		distances = []
		#print self.n_train
		# find the k nearest neighboor
		for i in range(self.n_train):
			distance = np.sum((test_data - self.train_data[i]) * (test_data - self.train_data[i])) #element wise multiplication
			distances.append(distance)

		ind = np.argsort(distances)
		cnt = np.zeros(10)
		for i in range(self.k):
			cnt[train_label[ind[i]]] += 1

		target = np.argmax(cnt)
		return target

knn = Knn_classifier(50)
knn.train(train_image, train_label)
cnt = 0
for i in range(1, num_test):
	t = knn.predict(test_image[i])
	#if i % 1000 == 0:
	if i % 100 == 0:
		print i
		print float(cnt) / i
		log.write('test size = ' + str(i) + ' test accuracy: ')
		log.write(str(float(cnt) / i))
		log.write('\n')
		#log.write('test size = ' + str(i) + '  test correctness: ')
		#log.write(str(float(cnt) / i)
		#log.write('\n')

	if t == test_label[i]:
		cnt += 1

print float(cnt) / num_test



		


