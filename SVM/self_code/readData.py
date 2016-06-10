import numpy as np 
import struct
import matplotlib.pyplot as plt
###############################
##read the data from the file##
###############################
trainImage = 'train-images-idx3-ubyte'
trainLabel = 'train-labels-idx1-ubyte'
testImage = 't10k-images-idx3-ubyte'
testLabel = 't10k-labels-idx1-ubyte'
def getImage(filename):
	image_file = open(filename,'rb')
	buff = image_file.read()
	image_file.close()
	bias = 0
	magic, sum, rol, col = struct.unpack_from('>IIII',buff,bias) #4BYTE*int INPUT,BIG ENDIAN
	print magic,' ',sum,' ',rol,' ',col
	bias += struct.calcsize('>IIII')
	#print bias
	images = [];
	for x in range(sum):
		im = struct.unpack_from('>784B',buff,bias) #28*28BIT
		bias += struct.calcsize('>784B')
		im = list(im)
		for i in range(len(im)):
			if im[i]>1:
				im[i] = 1
				#if a greyscale has the value bigger than 1, then we regard it as 1...nice accurate and less time
		images.append(im)
	return np.array(images) #get a vector 60000*784

def getLabel(filename):
	label_file = open(filename,'rb')
	buff = label_file.read()
	label_file.close()
	bias = 0
	magic, sum= struct.unpack_from('>II',buff,bias)  #2int INPUT, BIG ENDIAN
	print magic,' ',sum
	bias += struct.calcsize('>II')
	labels = [];
	for x in range(sum):
		im = struct.unpack_from('>1B',buff, bias)
		bias += struct.calcsize('>1B')
		labels.append(im[0])
	return np.array(labels) #the standard label vector,60000

def getTrainImage():
	return getImage(trainImage)

def getTestImage():
	return getImage(testImage)

def getTrainLabel():
	return getLabel(trainLabel)

def getTestLabel():
	return getLabel(testLabel)


