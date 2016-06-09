# number units of hidden layer in this version is 100
#network
#using vector to do all the calculations
#numpy is a good example

from read_map import *
from augment import *
import datetime
import math
import random
import struct
import numpy as np
import scipy

start_time = datetime.datetime.now()
train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()

log_file = 'log_data_aug_firstly'
log = open(log_file, 'w')

num_hid = 100 #number units of the hidden layer
num_fea = 784 #number of the feature vector which is the dimension of the input vector
num_out = 10  #number of the output layer 
num_images = 60000
num_test = 10000
decay_epoch = 10; #decay learning rate every 5 epoch
decay_rate = 0.1
alpha = 0.05 # learning speed 

def sigmoid(inx):
    return 1.0 / (1.0 + np.exp(- inx))
    
def softmax(z):
    #max_z = np.max(z)
    #tmp_z = z - max_z
    z = np.exp(z)
    Z = np.sum(z)
    return z / Z

def predict(X, y):
    count_true = 0.0
    count_t = 0.0
    # only do forward propagation
    for x in range( len(y) ):
        feature = X[x]
    
        feature = feature.reshape(num_fea,1)
        #hidden layer
        val_hid = np.dot(W_hid, feature)
        val_hid += b_hid
        sig_val_hid = sigmoid(val_hid)
        #output layer
        val_out = np.dot(W_out, sig_val_hid)
        val_out += b_out
        val_out = sigmoid(val_out)
        val_out = softmax(val_out)
        #print x
        #print val_out.reshape(10)
        i = 0
        max_ = np.max(val_out)
        #find the index of i
        for i in range(len(val_out)):
            if val_out[i] == max_:
                break
        #print val_out.reshape(1,10)
        #print i
        if i == y[x]:
            count_true += 1.0
    log.write(str(count_true / 1.0 / len(y)))
    log.write('\n')
    print count_true / 1.0 / len(y)

def predict_all(X):
    p = []
    for i in range(len(X)):
        feature = X[i]
        feature = feature.reshape(num_fea,1)
        #hidden layer
        val_hid = np.dot(W_hid, feature)
        val_hid += b_hid
        sig_val_hid = sigmoid(val_hid)
        #output layer
        val_out = np.dot(W_out, sig_val_hid)
        val_out += b_out
        val_out = sigmoid(val_out)
        m = np.argmax(val_out)
        p.append(m)
    return p

#intialize the parameters
#weight of the hidden layer
W_hid = np.random.rand( num_hid, num_fea)
W_hid -= 0.5
W_hid *= 0.1

#bias of the hidden layer
b_hid = np.random.rand( num_hid, 1)#bias of hidden unit 
b_hid -= 0.5
b_hid *= 0.1

#weight of the output layer
W_out = np.random.rand( num_out, num_hid)
W_out -= 0.5
W_out *= 0.1

#bias of the output layer
b_out = np.random.rand( num_out, 1 )#bias of out unit
b_out -= 0.5
b_out *= 0.1

total_num = len(train_image)

num_train = int(num_images*.8)
num_CV    = int(num_images)

np.random.seed(100)
np.random.shuffle(train_image)
np.random.seed(100)
np.random.shuffle(train_label)



X_train = train_image[:num_train]
X_CV    = train_image[num_train:num_CV]
X_test  = test_image

y_train = train_label[:num_train]
y_CV    = train_label[num_train:num_CV]
y_test  = test_label

X_train = reshape_image(X_train)
X_CV = reshape_image(X_CV)
X_test = reshape_image(X_test)

ori_train_x = X_train
ori_train_y = y_train
# update at the first time
print '==> start augmentation'
y_train, X_train, num_train = augment_images_ori(ori_train_x, ori_train_y)
print '==> end augmentation'
print num_train
print len(X_train)
'''
check OK!
print '==>start checking'
def check(src, src_label, dst, dst_label):
    n = len(dst)
    for i in range(n):
        #print str(i) + ' ',
        if dst_label[i] != src_label[i * 2]:
            print "wrong label"
        for j in range(28*28):
            if dst[i][j][0] != src[i * 2][j][0]:
                print 'diff pixels at', 
                print i
                break

check(X_train, y_train, ori_train_x, ori_train_y)
print '==> end checking'

'''

#Backpropagation
for ite in range(1, 100):#training times

    if ite % decay_epoch == 0:
        alpha *= decay_rate
    train_list = range(num_train)
    random.shuffle(train_list)
    for x in train_list:# train images

        #I used the stochastic gradient decent
        #forward propagation
        #print x
        feature = X_train[x] 
        #print feature.shape  
        feature = feature.reshape(num_fea,1)#change to column vector

        val_hid = np.dot(W_hid, feature)
        val_hid += b_hid
        sig_val_hid = sigmoid(val_hid)
   
        val_out = np.dot(W_out, sig_val_hid)
        val_out += b_out
        soft_val_out = softmax(val_out)
        
        target = y_train[x]
        
        #backword propagtion
        #calc the loglikelihood
        J = - np.log( soft_val_out[target] )
    
        #calc the gradient 
        dJ_dval_out = soft_val_out
        dJ_dval_out[target] -= 1.0
        sig_val_hid = sig_val_hid.reshape(1, num_hid)
        dJ_dW_out = np.dot(dJ_dval_out,sig_val_hid)
        dJ_dB_out = dJ_dval_out
        #dE_dval_out = np.multiply(dE_dval_out, soft_val_out)
        #dE_dval_out = np.multiply(dE_dval_out,1.0 - soft_val_out)
    
        dJ_dval_out = dJ_dval_out.reshape(1, num_out)#change to raw vector
        dJ_dsig_val_hid = np.dot(dJ_dval_out, W_out)
        
        dJ_dsig_val_hid = dJ_dsig_val_hid.reshape(num_hid, 1 )#change to column vector
        sig_val_hid = sig_val_hid.reshape(num_hid, 1)
        dsig_val_hid_dval_hid = np.multiply(sig_val_hid, 1 - sig_val_hid)
        dJ_dval_hid = np.multiply(dJ_dsig_val_hid, dsig_val_hid_dval_hid)
        
        feature = feature.reshape(1,num_fea)
        dJ_dW_hid = np.dot(dJ_dval_hid, feature)
        dJ_dB_hid = dJ_dval_hid
        
        #change the weights
        #W_out += b_alpha * b_out
        W_out -= alpha * dJ_dW_out
        #b_out = W_out100
        b_out -= alpha * dJ_dB_out
        #W_hid += b_alpha * b_hid
        W_hid -= alpha * dJ_dW_hid
        #b_hid = W_hid
        b_hid -= alpha * dJ_dB_hid
    print 'ite = ' , ite , '\n'

    if ite % 1 == 0:
        predict(X_train, y_train)
        log.write('---------\n')
        log.write('#ite  ')
        log.write(str(ite))
        log.write('\n')
        #print "-------"         
        #print "#ite", ite
        log.write('train correctness ')
        predict(ori_train_x, ori_train_y)
        log.write('CV correctness ')
        predict(X_CV,    y_CV)
        log.write('test correctness ')
        predict(X_test,  y_test)
    log.flush()
    
     # update the training set
    '''
    if ite  == 2:
        predict_labels = predict_all(ori_train_x)
        print '==> start augmentation'
        y_train, X_train, num_train = augment_images(ori_train_x, ori_train_y, predict_labels)
        print '==> end augmentation'
        print num_train
    '''