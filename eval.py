import numpy as np
from arff import Arff
from KNN import KNNClassifier

mat = Arff("./datasets/diabetes.arff",label_count=1)
mat2 = Arff("./datasets/diabetes_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance')
KNN.fit(train_data,train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data, test_labels)
print(score)
np.savetxt("diabetes.csv",pred,delimiter=',',fmt="%i")