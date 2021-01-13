import numpy as np
from arff import Arff
from KNN import KNNClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

mat = Arff("./datasets/housing.arff",label_count=1)
mat2 = Arff("./datasets/housing_test.arff", label_count=1)


# train
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:, :-1]
train_labels = raw_data[:, -1]
# test
raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:, -1]
# normalize
scaler = MinMaxScaler()
scaler.fit(train_data)
trans_train = scaler.transform(train_data)
trans_test = scaler.transform(test_data)


accs = []
ks = []
for k in range(1, 16, 2):
    KNNk = KNNClassifier(labeltype='regress', k=k)

    KNNk.fit(trans_train, train_labels)
    score = KNNk.score(trans_test, test_labels)

    accs.append(score)
    ks.append(k)
    


    

plt.plot(ks, accs)
plt.xlabel("K values")
plt.ylabel("MSE")
plt.title("housing: MSE with diff K values")
plt.show()