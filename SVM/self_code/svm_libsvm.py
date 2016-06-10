from svmutil import *
from readData import *
import datetime
start = datetime.datetime.now();
trainImage = getTrainImage()
trainLabel = getTrainLabel()
testImage = getTestImage()
testLabel = getTestLabel()

buff = []
print 'training:'
res = svm_train(buff, trainLabel, trainImage, '-c 4')

p_labs, p_acc, p_vals = svm_predict(testLabel,testImage, res)
true = 0
for i in range(len(testImage)):
    if(p_labs[i] ==testLabel[i]):
        true += 1
   # print i,' ',predictResult ,' ',testLabel[i]

end = datetime.datetime.now()
print 'error rate: '+ str(1-(true*1.0/len(testImage)))
