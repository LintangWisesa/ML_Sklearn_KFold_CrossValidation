import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
print(dir(digits))
# print(len(digits['data']))

from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    digits['data'], 
    digits['target'], 
    test_size = .1
)

# machine learning model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# cross validation
from sklearn.model_selection import cross_val_score

# print(cross_val_score(
#     LogisticRegression(),
#     xtr,
#     ytr
# ))
# print(cross_val_score(
#     SVC(gamma='auto'),
#     xtr,
#     ytr
# ))
# print(cross_val_score(
#     RandomForestClassifier(n_estimators=100),
#     xtr,
#     ytr
# ))

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