import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
print(dir(digits))
# print(len(digits['data']))

# machine learning model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# cross validation
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
k = StratifiedKFold(n_splits = 100)

# kfold
for train_index, test_index in k.split(digits['data'], digits['target']):
    xtr = digits['data'][train_index]
    ytr = digits['target'][train_index]
    
print(cross_val_score(
    LogisticRegression(),
    xtr,
    ytr
).mean())
print(cross_val_score(
    SVC(gamma='auto'),
    xtr,
    ytr
).mean())
print(cross_val_score(
    RandomForestClassifier(n_estimators=100),
    xtr,
    ytr
).mean())