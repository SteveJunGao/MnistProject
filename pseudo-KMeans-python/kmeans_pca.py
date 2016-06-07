# an implementation for KNN classifier
# I just need to calculate the distance between the test data and all of the training data

from read_map import *
from pca import *
import datetime
import math
import random
import struct
import numpy as np
import scipy
import heapq 

def pca(dataMat, topNfeat=100):  
    meanVals = np.mean(dataMat, axis=0)  
    meanRemoved = dataMat - meanVals 
    stded = meanRemoved / np.std(dataMat) # regulaization 
    covMat = np.cov(stded, rowvar=0) # calc the covariance 
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # calc the engin value of the matrix 
    eigValInd = np.argsort(eigVals)  #sort the engin value and find the max engin value  
    eigValInd = eigValInd[:-(topNfeat + 1):-1]    
    redEigVects = eigVects[:, eigValInd]       # reduce none important value
    lowDDataMat = stded * redEigVects    #calc the new matrix 
    #reconMat = (lowDDataMat * redEigVects.T) * std(dataMat) + meanVals  # calc the retrivale matrix
    return lowDDataMat

start_time = datetime.datetime.now()
train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()
num_images = 60000
num_test = 10000

 
log_file = 'log_kmeans_pca'
log = open(log_file, 'w')

class K_means_classifier:
	def __init__(self, k):
		self.k = k
	def train(self, train_data, train_label):
		# find the central point of each image_set
		image_set = []
		self.central_point = []
		for i in range(10):
			image_set.append([])
		n_train = len(train_data)
		# classify the training data
		# assume image with the same label is divided into the same set
		for i in range(n_train):
			image_set[train_label[i]].append(train_data[i])
		# calc the central point of each cluster
		image_set = np.array(image_set)
		for i in range(10):
			image_set[i] = np.array(image_set[i])
			#print image_set[i].shape
			n_set = len(image_set[i])
			sum_vec = image_set[i].sum(axis = 0)
			point = sum_vec / float(n_set)
			#print point.shape
			#print point
			self.central_point.append(point)
		self.central_point = np.array(self.central_point)
		#print len(self.central_point)

	def predict(self, test_data):
		distances = []
		#print test_data.shape
		for i in range(10):
			#print self.central_point[i].shape
			#element wise multiplication

			distance = np.inner(test_data -self.central_point[i], test_data - self.central_point[i])
			#print distance
			distance = np.sum(distance)
			distances.append(distance)
		distances = np.array(distances)
		#print distances
		t = np.argmin(distances)

		return t

k_means = K_means_classifier(10)
all_data = np.vstack((train_image, test_image))

all_data = pca(all_data, topNfeat = 700)
train_image = all_data[:60000]
test_image = all_data[60000:]

k_means.train(train_image, train_label)

#print k_means.central_point
cnt = 0
for i in range(1, num_test):
	t = k_means.predict(test_image[i])
	if t == test_label[i]:
		cnt += 1
	if i % 1000 == 0:
		print i 
		print float(cnt) / i
		log.write('test size = ' + str(i) + ' test accuracy: ')
		log.write(str(float(cnt) / i))
		log.write('\n')

print float(cnt) / i
