# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import time
import pandas as pd
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score, log_loss,roc_curve
from scipy.sparse import csr_matrix
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
# from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

# Where to save the figures
processed_dataset_dir = "deleted_dataset/"

#1. original dataset
original_set_train = joblib.load(processed_dataset_dir + 'original_set_train' + '.gz')
original_set_test = joblib.load(processed_dataset_dir + 'original_set_test' + '.gz')
original_labels_train = joblib.load(processed_dataset_dir + 'original_labels_train' + '.gz')
original_labels_test = joblib.load(processed_dataset_dir + 'original_labels_test' + '.gz')

original_set_train = csr_matrix.toarray(original_set_train)
original_set_test = csr_matrix.toarray(original_set_test)

# print(original_labels_test)
# 0 is majority, 1 is minority from original data
# 1 is inlier(majority), -1 is outlier(minority) from predict number

# dataset = np.concatenate((original_set_train,original_set_test))
# dataset_label = np.concatenate((original_labels_train,original_labels_test))

# dataset = joblib.load(processed_dataset_dir + 'raw_set' + '.gz')
# dataset_label = joblib.load(processed_dataset_dir + 'raw_labels' + '.gz')

# dataset = csr_matrix.toarray(dataset)

# #Boruta feature selection
# rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
# feat_selector.fit(dataset, dataset_label)
# dataset = feat_selector.transform(dataset)

# tree-based feature selection
# feature_clf = ExtraTreesClassifier(n_estimators=50)
# feature_clf = feature_clf.fit(dataset, dataset_label)
# tb_model = SelectFromModel(feature_clf, prefit=True)
# dataset = tb_model.transform(dataset)

# #L1-based feature selection
# #converge warning there
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter = 2000).fit(dataset, dataset_label)
# l_model = SelectFromModel(lsvc, prefit=True)
# dataset = l_model.transform(dataset)
# original_set_train, original_set_test, original_labels_train, original_labels_test = train_test_split(dataset,dataset_label, test_size=0.2, random_state=42)

# print(original_set_train[0])
# exit(0)

# clean training dataset
clean_train_dataset = []
# novelty = []
# novelty_label = []
for i in range(original_labels_train.shape[0]):
    if original_labels_train[i] == 0:
        clean_train_dataset.append(original_set_train[i])
    # else:
    #     novelty.append(original_set_train[i])
    #     novelty_label.append(original_labels_train[i])

# clean_train_dataset = np.array(clean_train_dataset)
# novelty = np.array(novelty)

clean_test_dataset = []
novelty = []
novelty_label = []
for i in range(original_labels_test.shape[0]):
    if original_labels_test[i] == 0:
        clean_test_dataset.append(original_set_test[i])
    else:
        novelty.append(original_set_test[i])
        novelty_label.append(original_labels_test[i])

clean_test_dataset = np.array(clean_test_dataset)
novelty = np.array(novelty)

# print("original_set_train_shape:",original_set_train.shape[0])
# print("train_dataset_shape:",clean_train_dataset.shape[0])


# one class learning there
# clf = OneClassSVM(kernel='rbf',gamma='auto').fit(original_set_train)
# prediction = clf.predict(original_set_test)
# clf = OneClassSVM(kernel='rbf',gamma='auto').fit(clean_train_dataset)
# prediction = clf.predict(novelty)
# clf = OneClassSVM(kernel='rbf',gamma='auto').fit(original_set_train)
# prediction = clf.predict(original_set_test)

# prediction = OneClassSVM(gamma='auto').fit_predict(dataset)
# prediction = EllipticEnvelope(random_state=0).fit_predict(dataset)
# prediction = IsolationForest().fit_predict(dataset)

clf = OneClassSVM(gamma='auto').fit(original_set_train)
prediction = clf.predict(original_set_test)

total_num = prediction.shape[0]

# transfer prediction to the same value metric
# print("prediction",prediction)
for i in range(total_num):
    if prediction[i] == -1:
        prediction[i] = 1
    else:
        prediction[i] = 0
# print("prediction",prediction)

minority_predict = 0
minority_sum = 0
for i in range(total_num):
    if original_labels_test[i] == 1:
        minority_sum += 1
        if prediction[i] == 1:
            minority_predict += 1

print("ratio:",minority_predict/minority_sum)

# true positive, true negative, false positive, false negative
# 0 is majority, 1 is minority
# TP = 0
# TN = 0
# FP = 0
# FN = 0

# for i in range(total_num):
#     if original_labels_test[i] == 1:
#         if prediction[i] == 1:
#             TP += 1
#         else:
#             FN += 1
#     else:
#         if prediction[i] == 1:
#             FP += 1
#         else:
#             TN += 1
# f = open("outlier_detection_result.txt","w+")

# print("TP:",TP)
# print("FN:",FN)
# print("FP:",FP)
# print("TN:",TN)

# # #precision: how many correct in your prediction
# print("precision:",TP/(TP+FP))
# # #recall: how many correct in real minority
# print("recall:",TP/(TP+FN))
# print("accuracy:",(TP+TN)/total_num)

# f.write("TP:"+str(TP)+"\r\n")
# f.write("FN:"+str(FN)+"\r\n")
# f.write("FP:"+str(FP)+"\r\n")
# f.write("TN:"+str(TN)+"\r\n")

# #precision: how many correct in your prediction
# f.write("precision:"+str(TP/(TP+FP))+"\r\n")
# #recall: how many correct in real minority
# f.write("recall:"+str(TP/(TP+FN))+"\r\n")
# f.write("accuracy:"+str((TP+TN)/total_num)+"\r\n")

# f.close()


