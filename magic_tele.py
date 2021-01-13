import numpy as np
from arff import Arff
from KNN import KNNClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

mat = Arff("./datasets/magic_telescope.arff",label_count=1)
mat2 = Arff("./datasets/magic_telescope_test.arff", label_count=1)


raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:, :-1]
train_labels = raw_data[:, -1]




raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:, -1]

scaler = MinMaxScaler()
scaler.fit(train_data)
trans_train = scaler.transform(train_data)
trans_test = scaler.transform(test_data)

'''KNN2 = KNNClassifier(labeltype='classification')
KNN2.fit(train_data, train_labels)
score2 = KNN2.score(test_data, test_labels)
print('part 2a\n')
print(f'k= 3 wout/normalization Acc = {score2}')



KNN = KNNClassifier(labeltype='classification')
KNN.fit(trans_train, train_labels)
score = KNN.score(trans_test, test_labels)

print(f'k= 3 w/normalization Acc = {score}')'''



print('\npart 2b\n')

accs = []
ks = []
for k in range(1, 16, 2):
    KNNk = KNNClassifier(labeltype='classification', k=k)

    KNNk.fit(trans_train, train_labels)
    score = KNNk.score(trans_test, test_labels)

    accs.append(score)
    ks.append(k)

    

plt.plot(ks, accs)
plt.xlabel("K values")
plt.ylabel("Acurracy")
plt.title("Magic Telescope: Accuracy with diff K values")
plt.show()