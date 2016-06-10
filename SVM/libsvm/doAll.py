from svmutil import *
from svm import *
from read_map import *
import numpy as np
import codecs
import time

#start_time = datetime.datetime.now()
train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()
num_images = 60000
num_test = 10000
start_tarin_t = time.clock()
print '==> training'
print len(train_image)
#print train_image[0]
W = []
model = svm_train(W, train_label, train_image, '-c 5')
print '==> finished training'
print 'Time Elasping for training : ',
end_train_t = time.clock()
print end_train_t
print '==> predicting'
p_labs, p_acc, p_vals = svm_predict(test_label, test_image, model )
print 'finished predicting'
print 'Time Elasping for predicting : '
end_predict_t = time.clock()
t_predict = end_predict_t - end_train_t
print t_predict
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
