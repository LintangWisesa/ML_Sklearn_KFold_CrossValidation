# K-Fold cross validation
# which model has the best performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
print(dir(digits))
# print(len(digits['data']))

# split: 90% train & 10% test
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    digits['data'], 
    digits['target'], 
    test_size = .1
)
print(len(xtr))
print(len(xts))

# machine learning model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# function get score manually
def get_score(model, xtr, xts, ytr, yts):
    model.fit(xtr, ytr)
    return model.score(xts, yts)

# print(get_score(LogisticRegression(), xtr, xts, ytr, yts))
# print(get_score(SVC(gamma='auto'), xtr, xts, ytr, yts))
# print(get_score(RandomForestClassifier(n_estimators = 100), xtr, xts, ytr, yts))

# manual k-fold cross validation
from sklearn.model_selection import KFold
k = KFold(n_splits = 3)     # k max: n data in dataset

# mylist = [0,1,2,3,4,5,6,7,8,9]
# for train_index, test_index in k.split(mylist):
#     print(train_index, test_index)

skorLogreg = []
skorSVC = []
skorRanfor = []

for train_index, test_index in k.split(digits['data']):
    xtr = digits['data'][train_index]
    xts = digits['data'][test_index]
    ytr = digits['target'][train_index]
    yts = digits['target'][test_index]

    # print(get_score(LogisticRegression(), xtr, xts, ytr, yts))
    # print(get_score(SVC(gamma='auto'), xtr, xts, ytr, yts))
    # print(get_score(RandomForestClassifier(n_estimators = 100), xtr, xts, ytr, yts))

    skorLogreg.append(get_score(LogisticRegression(), xtr, xts, ytr, yts))
    skorSVC.append(get_score(SVC(gamma='auto'), xtr, xts, ytr, yts))
    skorRanfor.append(get_score(RandomForestClassifier(n_estimators = 100), xtr, xts, ytr, yts))

print(skorLogreg)
print(skorSVC)
print(skorRanfor)

# print mean score
print(np.array(skorLogreg).mean())
print(np.array(skorSVC).mean())
print(np.array(skorRanfor).mean())
