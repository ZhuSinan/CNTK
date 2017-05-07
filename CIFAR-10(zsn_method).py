import numpy as np
import os
from sklearn import svm


train_file = os.path.join('data', 'CIFAR-10', 'Train_cntk_text.txt')
test_file = os.path.join('data', 'CIFAR-10', 'Test_cntk_text.txt')

f = open(train_file, 'r')

train_features = []
train_labels = []
for line in f.readlines():
    line = line.strip().split()
    del(line[11], line[0])
    line = list(map(lambda x : int(x), line))
    train_features.append(line[10:])
    train_labels.append(np.argmax(line[:10]))
f.close()

f = open(test_file)
test_features = []
test_labels = []
for line in f.readlines():
    line = line.strip().split()
    del(line[11], line[0])
    line = list(map(lambda x : int(x), line))
    test_features.append(line[10:])
    test_labels.append(np.argmax(line[:10]))
f.close()

clf = svm.SVC()
clf.fit(train_features,train_labels)

pred = clf.predict(test_features)
accuracy = 0
for index in range(len(test_labels)):
    if test_labels[index] == pred[index]:
        accuracy += 1

accuracy = accuracy/len(test_labels)
print(accuracy)


