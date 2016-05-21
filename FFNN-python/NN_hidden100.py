# number units of hidden layer in this version is 100
#network
#using vector to do all the calculations
#numpy is a good example

from read_map import *
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
 
log_file = 'log_hidden100'
log = open(log_file, 'w')

num_hid = 100 #number units of the hidden layer
num_fea = 784 #number of the feature vector which is the dimension of the input vector
num_out = 10  #number of the output layer 
num_images = 60000
num_test = 10000
alpha = 0.005# learning speed 
b_alpha = 0.0001

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
    for x in range( len(y) ):
        feature = X[x]
    
        feature = feature.reshape(num_fea,1)
   
        val_hid = np.dot(W_hid, feature)

        sig_val_hid = sigmoid(val_hid)
   
        val_out = np.dot(W_out, sig_val_hid)
        val_out = sigmoid(val_out)
        i = 0
        max_ = np.max(val_out)
        for i in range(len(val_out)):
            if val_out[i] == max_:
                break
        #print val_out.reshape(1,10)
        #print i
        if i == y[x]:
            count_true += 1.0
    log.write(str(count_true / 1.0 / len(y)))
    log.write('\n')
    #print count_true / 1.0 / len(y)
    

W_hid = np.random.rand( num_hid, num_fea)
W_hid -= 0.5
W_hid *= 0.1


b_hid = np.random.rand( num_hid, 1)#bias of hidden unit 
b_hid -= 0.5
b_hid *= 0.1

W_out = np.random.rand( num_out, num_hid)
W_out -= 0.5
W_out *= 0.1


b_out = np.random.rand( num_out, 1 )#bias of out unit
b_out -= 0.5
b_out *= 0.1

total_num = len(train_image)

num_train = int(num_images*.6)
num_CV    = int(num_images*.8)

np.random.seed(100)
np.random.shuffle(train_image)
np.random.seed(100)
np.random.shuffle(train_label)



X_train = train_image[:num_train]
X_CV    = train_image[num_train:num_CV]
X_test  = train_image[num_CV:]

y_train = train_label[:num_train]
y_CV    = train_label[num_train:num_CV]
y_test  = train_label[num_CV:]


#Backpropagation
for ite in range(100):#training times

    for x in range(num_train):# train images

    
        feature = X_train[x]   
        feature = feature.reshape(num_fea,1)#change to column vector

        val_hid = np.dot(W_hid, feature)

        sig_val_hid = sigmoid(val_hid)
   
        val_out = np.dot(W_out, sig_val_hid)

        soft_val_out = softmax(val_out)
        
        target = y_train[x]
        
        J = - np.log( soft_val_out[target] )
    
        
        dJ_dval_out = soft_val_out
        dJ_dval_out[target] -= 1.0
        sig_val_hid = sig_val_hid.reshape(1, num_hid)
        dJ_dW_out = np.dot(dJ_dval_out,sig_val_hid)
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
        
        #W_out += b_alpha * b_out
        W_out -= alpha * dJ_dW_out
        #b_out = W_out100
        
        #W_hid += b_alpha * b_hid
        W_hid -= alpha * dJ_dW_hid
        #b_hid = W_hid
    print 'ite = ' , ite , '\n'

    if ite % 2 == 0:
        log.write('---------\n')
        log.write('#ite  ')
        log.write(str(ite))
        log.write('\n')
        #print "-------"         
        #print "#ite", ite
        log.write('train correctness ')
        predict(X_train, y_train)
        log.write('CV correctness ')
        predict(X_CV,    y_CV)
        log.write('test correctness ')
        predict(X_test,  y_test)
    log.flush()