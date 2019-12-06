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
from sklearn.svm import LinearSVC
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


classifier = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)
y_score = classifier.fit(original_set_train, original_labels_train).decision_function(original_set_test)

fpr, tpr, thresholds = metrics.roc_curve(original_labels_test, y_score)

roc_auc = roc_auc_score(original_labels_test, y_score)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LinearSVC ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()