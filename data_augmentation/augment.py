from read_map import *
import numpy as np
import random
import matplotlib.pyplot as plt


def left_shift(image, n):
	a = np.zeros([28,28])
	for i in range(28 - n):
		a[i + n][i] = 1
	return np.dot(image, a)

def right_shift(image, n):
	a = np.zeros([28,28])
	for i in range(28 - n):
		a[i][i + n] = 1
	return np.dot(image, a)


def up_shift(image, n):
	a = np.zeros([28,28])
	for i in range(28 - n):
		a[i][i + n] = 1
	return np.dot(a, image)

def down_shift(image, n):
	a = np.zeros([28,28])
	for i in range(28 - n):
		a[i + n][i] = 1
	return np.dot(a, image)

def augment_images_ori(images, labels):
	n = len(images)
	new_train_images = []
	new_train_labels = []
	new_num = 0
	for i in range(n):
		image = images[i]
		#image = image.reshape(28, 28).reshape(28*28, 1)
		new_train_images.append(image)
		image = image.reshape(28,28)
		# random augment images
		t = random.randint(1, 4)
		if t == 1:
			new_image = left_shift(image, 1).reshape(28*28, 1)
		if t == 2:
			new_image = right_shift(image, 1).reshape(28*28, 1)
		if t == 3:
			new_image = up_shift(image, 1).reshape(28*28, 1)
		if t == 4:
			new_image = down_shift(image, 1).reshape(28*28, 1)
		new_train_images.append(new_image)
		# augment the labels at the same time
		for j in range(2):
			new_train_labels.append(labels[i])
		new_num += 2
	return new_train_labels, new_train_images, new_num

def reshape_image(images):
	n = len(images)
	new_train_images = []
	for i in range(n):
		image = images[i]
		image = image.reshape(28 , 28).reshape(28*28, 1)
		new_train_images.append(image)
	return new_train_images
#I only augment images that isn't predicted correctly

def augment_images_random(images, labels, n):
	num_image = len(labels)
	aug_list = range(num_image)
	random.shuffle(aug_list)
	aug_list = aug_list[:n]
	new_train_images = []
	new_train_labels = []
	new_num = 0
	np.sort(aug_list)
	for i in range(num_image):
		image = images[i]
		new_train_images.append(image)
		new_train_labels.append(labels[i])
		new_num += 1
		if i in aug_list:
			new_train_images.append(image)
			new_train_labels.append(labels[i])
			new_num += 1
	return new_train_labels, new_train_images, new_num


def augment_images(images, labels, predict_labels):
	n = len(images)
	print n
	new_train_images = []
	new_train_labels = []
	new_num = 0
	for i in range(n):
		if labels[i] == predict_labels[i]:
			new_train_images.append(images[i])
			new_train_labels.append(labels[i])
			new_num += 1
			continue
		else:
			#augment the original images 2 times
			new_train_images.append(images[i])
			#new_train_images.append(images[i])
			# augment the image at 4 different positions
			image = images[i]
			image = image.reshape(28,28)
			# random augment images
			t = random.randint(1, 4)
			if t == 1:
				new_image = left_shift(image, 1).reshape(28*28, 1)
			if t == 2:
				new_image = right_shift(image, 1).reshape(28*28, 1)
			if t == 3:
				new_image = up_shift(image, 1).reshape(28*28, 1)
			if t == 4:
				new_image = down_shift(image, 1).reshape(28*28, 1)
			new_train_images.append(new_image)
			# augment the labels at the same time
			for j in range(2):
				new_train_labels.append(labels[i])
			new_num += 2
			
	return new_train_labels, new_train_images, new_num


def test_augment():
	train_image = get_train_images()
	im = train_image[0]
	print im
	im = train_image[0].reshape(28*28, 1)
	print im
	im = train_image[0].reshape(28, 28)
	print im
	im = im.reshape(28*28, 1)
	print im

	plt.imshow(im, cmap = 'binary')
	plt.show()
	im = left_shift(im, 2)
	plt.imshow(im, cmap = 'binary')
	plt.show()
	im = right_shift(im, 2)
	plt.imshow(im, cmap = 'binary')
	plt.show()
	im = up_shift(im, 2)
	plt.imshow(im, cmap = 'binary')
	plt.show()
	im = down_shift(im, 2)
	plt.imshow(im, cmap = 'binary')
	plt.show()
	im = left_shift(im, 2)
#test_augment()
