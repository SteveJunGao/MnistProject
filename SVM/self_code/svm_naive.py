#__author__ = 'liufangchen'
# ten double -classifier
#use handwritten_svm
from readData import *
import datetime
import handwritten_svm
import numpy as py
start = datetime.datetime.now();
trainImage = getTrainImage()
trainLabel = getTrainLabel()
testImage = getTestImage()
testLabel = getTestLabel()


##########
# use PCA DECREASE THE DIMENSION
##########
def pca(dataMat, topNfeat=100):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    stded = meanRemoved / np.std(dataMat) # regulaization
    covMat = np.cov(stded, rowvar=0) # calc the covariance
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # calc the engin value of the matrix
    eigValInd = np.argsort(eigVals) #sort the engin value and find the max engin value
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd] # reduce none important value
    lowDDataMat = stded * redEigVects #calc the new matrix
    #reconMat = (lowDDataMat * redEigVects.T) * std(dataMat) + meanVals # calc the retrivale matrix
    return lowDDataMa

#label0 = open('label0.txt','w')
#label1 = open('label1.txt','w')
#label2 = open('label2.txt','w')
#label3 = open('label3.txt','w')
#label4 = open('label4.txt','w')
#label5 = open('label5.txt','w')
#label6 = open('label6.txt','w')
#label7 = open('label7.txt','w')
#label8 = open('label8.txt','w')
#label9 = open('label9.txt','w')

#tlabel0 = open('tlabel0.txt','w')
#tlabel1 = open('tlabel1.txt','w')
#tlabel2 = open('tlabel2.txt','w')
#tlabel3 = open('tlabel3.txt','w')
#tlabel4 = open('tlabel4.txt','w')
#tlabel5 = open('tlabel5.txt','w')
#tlabel6 = open('tlabel6.txt','w')
#tlabel7 = open('tlabel7.txt','w')
#tlabel8 = open('tlabel8.txt','w')
#tlabel9 = open('tlabel9.txt','w')
a = py.zeros((60000, 10))
b = py.zeros((10000, 10))
for i in range(60000):
    for j in range(10):
        if (trainLabel[i] == j):
            a[i,j] = 1
        else:
            a[i,j] = -1
for i in range(10000):
    for j in range(10):
        if (trainLabel[i] == j):
            b[i,j] = 1
        else:
            b[i,j] = -1
#label0.write(str(a[:, 0]))
#label1.write(str(a[:, 1]))
##label2.write(str(a[:, 2]))
#label3.write(str(a[:, 3]))
#label4.write(str(a[:, 4]))
#label5.write(str(a[:, 5]))
#label6.write(str(a[:, 6]))
#label7.write(str(a[:, 7]))
#label8.write(str(a[:, 8]))
#label9.write(str(a[:, 9]))

#tlabel0.write(str(b[:, 0]))
#tlabel1.write(str(b[:, 1]))
#tlabel2.write(str(b[:, 2]))
#tlabel3.write(str(b[:, 3]))
#tlabel4.write(str(b[:, 4]))
#tlabel5.write(str(b[:, 5]))
#tlabel6.write(str(b[:, 6]))
#tlabel7.write(str(b[:, 7]))
#tlabel8.write(str(b[:, 8]))
#tlabel9.write(str(b[:, 9]))

#label0.close()
#label1.close()
#label2.close()
#label3.close()
#label4.close()
#label5.close()
#label6.close()
#label7.close()
#label8.close()
#label9.close()

#tlabel0.close()
#tlabel1.close()
##tlabel2.close()
#tlabel3.close()
#tlabel4.close()
#tlabel5.close()
#tlabel6.close()
#tlabel7.close()
#tlabel8.close()
#tlabel9.close()

print ("step 1: load data...")
data = py.mat(trainImage)
train_x = data[:]
train_pca = pca(data,300).real
train_y0 = py.mat(a[:,0]).T
train_y1 = py.mat(a[:,1]).T
train_y2 = py.mat(a[:,2]).T
train_y3 = py.mat(a[:,3]).T
train_y4 = py.mat(a[:,4]).T
train_y5 = py.mat(a[:,5]).T
train_y6 = py.mat(a[:,6]).T
train_y7 = py.mat(a[:,7]).T
train_y8 = py.mat(a[:,8]).T
train_y9 = py.mat(a[:,9]).T

test_x = py.mat(testImage)
test_y0 = py.mat(b[:,0]).T
test_y1 = py.mat(b[:,1]).T
test_y2 = py.mat(b[:,2]).T
test_y3 = py.mat(b[:,3]).T
test_y4 = py.mat(b[:,4]).T
test_y5 = py.mat(b[:,5]).T
test_y6 = py.mat(b[:,6]).T
test_y7 = py.mat(b[:,7]).T
test_y8 = py.mat(b[:,8]).T
test_y9 = py.mat(b[:,9]).T

## step 2: training...
print "training class 0"
print ("step 2: training...")
C = 0.1
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y0, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y0)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 1"
print ("step 2: training...")
C = 3
toler = 0.0001
Iter = 2
svmClassifier = handwritten_svm.trainSVM(train_x, train_y1, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y1)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 2"
print ("step 2: training...")
C = 0.1
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y2, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y2)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 3"
print ("step 2: training...")
C = 1.2
toler = 0.0001
Iter = 2
svmClassifier = handwritten_svm.trainSVM(train_x, train_y3, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y3)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 4"
print ("step 2: training...")
C = 2
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y4, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y4)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 5"
print ("step 2: training...")
C = 2
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y5, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y5)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 6"
print ("step 2: training...")
C = 3
toler = 0.001
Iter = 2
svmClassifier = handwritten_svm.trainSVM(train_x, train_y6, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y6)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 7"
print ("step 2: training...")
C = 2
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y7, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y7)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 8"
print ("step 2: training...")
C = 2
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y8, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y8)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))

print "training class 9"
print ("step 2: training...")
C = 3
toler = 0.0001
Iter = 1
svmClassifier = handwritten_svm.trainSVM(train_x, train_y9, C, toler, Iter, kernelOption = ('rbf', 1.0))

## step 3: testing
print ("step 3: testing...")
accuracy = handwritten_svm.test_accuracy(svmClassifier, test_x, test_y9)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))


print 'end'