# manual cross validation
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

# compare accuracy logistic regression vs support vector machine vs random forest

# score logistic regression
from sklearn.linear_model import LogisticRegression
mLogreg = LogisticRegression()
mLogreg.fit(xtr, ytr)
skorLogreg = mLogreg.score(xts, yts)
print('Akurasi LogReg =', skorLogreg * 100, '%')

# score support vector machine
from sklearn.svm import SVC
mSVC = SVC(gamma = 'auto')
mSVC.fit(xtr, ytr)
skorSvc = mSVC.score(xts, yts)
print('Akurasi SVC =', skorSvc * 100, '%')

# score random forest
from sklearn.ensemble import RandomForestClassifier
mRanfor = RandomForestClassifier(
    n_estimators = 100
)
mRanfor.fit(xtr, ytr)
skorRF = mRanfor.score(xts, yts)
print('Akurasi RanFor =', skorRF * 100, '%')