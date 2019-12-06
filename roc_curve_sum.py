print(__doc__)

from numpy import array
from numpy import argmax
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import os
import time
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from skrules import SkopeRules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from scipy.sparse import csr_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from itertools import cycle
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

processed_dataset_dir = "deleted_dataset/"


#1. original dataset
original_set_train = joblib.load(processed_dataset_dir + 'original_set_train' + '.gz')
original_set_test = joblib.load(processed_dataset_dir + 'original_set_test' + '.gz')
original_labels_train = joblib.load(processed_dataset_dir + 'original_labels_train' + '.gz')
original_labels_test = joblib.load(processed_dataset_dir + 'original_labels_test' + '.gz')

original_set_train = csr_matrix.toarray(original_set_train)
original_set_test = csr_matrix.toarray(original_set_test)

classifier_list = [ \
    DecisionTreeClassifier(), \
    LinearSVC(random_state=0, tol=1e-5,max_iter=20000), \
    neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance'), \
    GaussianNB(), \
    SkopeRules(), \
    RandomForestClassifier(n_estimators=100, random_state=0), \
    AdaBoostClassifier(n_estimators=100), \
    VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), \
                            ('knn', neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')), \
                            ('gnb', GaussianNB())], voting='soft')\
]

model_name_list = ["decision_tree", "linear_svc", "knn", "gaussianNB", "skope_rule", "random_forest", "boosting", "hybrid"]

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

plt.figure()
lw = 2

for i in range(len(classifier_list)):
    if i is not 1 and i is not 4:
        y_score = classifier_list[i].fit(original_set_train, original_labels_train).predict_proba(original_set_test)
        y_score = y_score[:, 1]
    else:
        y_score = classifier_list[i].fit(original_set_train, original_labels_train).decision_function(original_set_test)

    # fpr, tpr, thresholds = metrics.roc_curve(original_labels_test, y_score)
    roc_auc = roc_auc_score(original_labels_test, y_score)
    fpr, tpr, thresholds = roc_curve(original_labels_test, y_score)

    # roc_auc = roc_auc_score(original_labels_test, y_score)

    plt.plot(fpr, tpr, color=color_list[i],
            lw=lw, label=model_name_list[i]+' (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve_sum.png")
plt.show()