import numpy as ny
import time
import math

# testing your trained svm model given test set
def test_accuracy(svm, test_x, test_y):
	test_x = ny.mat(test_x)
	test_y = ny.mat(test_y)
	numTestsamples = test_x.shape[0]
	SVIndex = ny.nonzero(svm.alphas.A > 0)[0]
	SVectors 		= svm.train_x[SVIndex]
	SVLabels = svm.train_y[SVIndex]
	SVAlphas = svm.alphas[SVIndex]
	true = 0
	for i in xrange(numTestsamples):
		kernelValue = calcKernel(SVectors, test_x[i, :], svm.kernel_kind)
		predict = kernelValue.T * (ny.multiply(SVLabels, SVAlphas)) + svm.b
		if predict * test_y[i] > 0:
			true += 1
	accuracy = float(true) / numTestsamples
	return accuracy


####################################################
# complete SVM
####################################################
def calcKernel(matrix_x, sample_x, kernelkind):
	numsample = matrix_x.shape[0]
	kernel_val = ny.mat(ny.zeros((numsample,1)))

	if kernelkind[0] == 'rbf':
		sigma = kernelkind[1]
		if sigma == 0:
			sigma = 1.0
		for i in xrange(numsample):
			diff = matrix_x[i, :] - sample_x
			kernel_val[i] = math.exp(diff * diff.T / (-2.0 * sigma**2))
	if kernelkind[0] == 'linear':
		kernel_val = matrix_x * sample_x.T

	return kernel_val



# calculate kernel matrix
def calcKernelMatrix(train_x, kernelkind):
	numsamples = train_x.shape[0]
	kernelMatrix = ny.mat(ny.zeros((numsamples, numsamples)))
	for i in xrange(numsamples):
		kernelMatrix[:, i] = calcKernel(train_x, train_x[i, :], kernelkind)
	return kernelMatrix


class SVMStruct:
	def __init__(self, dataSet, labels, C, toler, kernelkind):
		self.train_x = dataSet # each row stands for a sample
		self.train_y = labels  # corresponding label
		self.C = C             # slack variable
		self.toler = toler     # termination condition for iteration
		self.numSamples = dataSet.shape[0] # number of samples
		self.alphas = ny.mat(ny.zeros((self.numSamples, 1))) # Lagrange factors: vector alpha
		self.error_mem = ny.mat(ny.zeros((self.numSamples, 2)))
		self.b = 0
		self.kernel_kind = kernelkind
		self.kernelMatrix = calcKernelMatrix(self.train_x, self.kernel_kind)



# after optimizing alpha_k, we should refresh the state of error
def updateError(svm, alpha_k):
	error = calError(svm, alpha_k)
	svm.error_mem[alpha_k] = [1, error]

# calculate the error for alpha k
def calError(svm, alpha_k):
	output_k = float(ny.multiply(svm.alphas, svm.train_y).T * svm.kernelMatrix[:, alpha_k] + svm.b)
	error_k = output_k - float(svm.train_y[alpha_k])
	return error_k


# select alpha j which will maximum E_i - E_j, the second factor we will choose
# use global memory to store all of the deviation, so we needn't to calculate them all the time in the loop

def selectAlpha_j(svm, alpha_i, error_i):
	svm.error_mem[alpha_i] = [1, error_i]
	factorList = ny.nonzero(svm.error_mem[:, 0].A)[0]
	max = 0; alpha_j = 0; error_j = 0

	# find the alpha j to maximum error_i - error_j
	if len(factorList) > 1:
		for alpha_k in factorList:
			if alpha_k == alpha_i:
				continue
			error_k = calError(svm, alpha_k)
			if abs(error_k - error_i) > max:
				max = abs(error_k - error_i)
				alpha_j = alpha_k
				error_j = error_k

	#if the list has nothing , then we select it randomly
	else:
		alpha_j = alpha_i
		while alpha_j == alpha_i:
			alpha_j = int(ny.random.uniform(0, svm.numSamples))
		error_j = calError(svm, alpha_j)
	return alpha_j, error_j


# check and pick up the alpha which doesn't suit the KKT condition
	# KKT condition:
	#  yi*f(i) >= 1 and alpha == 0
	#  yi*f(i) == 1 and 0<alpha< C
	#  yi*f(i) <= 1 and alpha == C

	#  y[i]*error_i = y[i] * f(i) - y[i]^2 = y[i]*f(i) - 1, so
	# 1) if y[i]*error_i < 0, alpha < c
	# 2) if y[i]*error_i > 0, alpha > 0
	# 3) if y[i]*error_i = 0, so yi*f(i) = 1, it is on the boundary

def optim_ij(svm, alpha_i):
	error_i = calError(svm, alpha_i)

	if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or\
		(svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

		# select alpha j
		alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
		alpha_i_old = svm.alphas[alpha_i].copy()
		alpha_j_old = svm.alphas[alpha_j].copy()

		# calculate the boundary L and H
		if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
			Low = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
			High = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
		else:
			Low = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
			High = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
		if Low == High:
			return 0

		# calculate the similarity of sample i and j
		likely = 2.0 * svm.kernelMatrix[alpha_i, alpha_j] - svm.kernelMatrix[alpha_i, alpha_i] \
				  - svm.kernelMatrix[alpha_j, alpha_j]
		if likely >= 0:
			return 0


		svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / likely


		if svm.alphas[alpha_j] > High:
			svm.alphas[alpha_j] = High
		if svm.alphas[alpha_j] < Low:
			svm.alphas[alpha_j] = Low

		# if alpha j not moving enough, just return
		if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
			updateError(svm, alpha_j)
			return 0

		# update alpha_i
		svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
								* (alpha_j_old - svm.alphas[alpha_j])

		# update b (threshold value)
		# b1 =  b - error_i -  y^i (a_i - a_i_old)<xi,xi> - yj(a_j - a_j_old)<xi,xj>
		#b2 = b - error_j -  y^i (a_i - a_i_old)<xi,xj> - yj(a_j - a_j_old)<xj,xj>

		b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)\
													* svm.kernelMatrix[alpha_i, alpha_i]\
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)\
													* svm.kernelMatrix[alpha_i, alpha_j]
		b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)\
													* svm.kernelMatrix[alpha_i, alpha_j]\
							 - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old)\
													* svm.kernelMatrix[alpha_j, alpha_j]
		if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
			svm.b = b1
		elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
			svm.b = b2
		else:
			svm.b = (b1 + b2) / 2.0

		# update the global cache when it is complete
		updateError(svm, alpha_j)
		updateError(svm, alpha_i)

		return 1
	else:
		return 0

# the main training procedure
def trainSVM(train_image, train_label, C, toler, maxIter, kernelOption = ('rbf', 1.0)):
	svm = SVMStruct(ny.mat(train_image), ny.mat(train_label), C, toler, kernelOption)
	flag = True
	ifChanged = 0 #to mark how many alphas is changed
	iter = 0
	while (iter < maxIter) and ((ifChanged > 0) or flag):
		ifChanged = 0

		if flag:
			#update all the alphas
			for i in xrange(svm.numSamples):
				ifChanged += optim_ij(svm, i)
			iter += 1 #update the round

		else:
			# update alphas if it is not zero or c, and update ifchanged
			alpha_update = ny.nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
			for i in alpha_update:
				ifChanged += optim_ij(svm, i)
			iter += 1 #update

		if flag == True:
			flag = False
		elif ifChanged == 0:
			flag = True
	print 'Training complete!'
	return svm




