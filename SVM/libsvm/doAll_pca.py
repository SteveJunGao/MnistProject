from svmutil import *
from svm import *
from read_map import *
import numpy as np
import codecs
import time

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

#start_time = datetime.datetime.now()
train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()
num_images = 60000
num_test = 10000
print time.clock()
all_data = np.vstack((train_image, test_image))
print '==> start PCA'
all_data = pca(all_data, topNfeat = 390)
print '==> end pca'
print 'Time Elaspsing for PCA: ',
print time.clock()
all_data = all_data.real
train_image = all_data[:60000]
test_image = all_data[60000:]
#print train_image[0]
train_image = train_image.tolist()
test_image = test_image.tolist()

print '==> training'
print len(train_image)
#print train_image[0]
W = []
model = svm_train(W, train_label, train_image, '-c 5')
print '==> finished training'
print ' Time Elasping for training : ',
print time.clock()
print '==> predicting'
p_labs, p_acc, p_vals = svm_predict(test_label, test_image, model )
print 'finished predicting'
print 'Time Elasping for predicting : '
print time.clock()
cnt_true = 0

f = codecs.open('svm_result_pca_p.csv', 'w', 'utf-8')
for i in range(num_test):
	for j in range(10):
		f.write(str(p_vals[i][j]) + ',')
	f.write('\n')

for i in range(num_test):
	if p_labs[i] == test_label[i]:
		cnt_true += 1

print float(cnt_true) / num_test
