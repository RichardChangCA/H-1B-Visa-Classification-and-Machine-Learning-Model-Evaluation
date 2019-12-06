# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

#python3 -m memory_profiler classification.py

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from skrules import SkopeRules
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from scipy.sparse import csr_matrix

# Where to save the figures
processed_dataset_dir = "deleted_dataset/"

#load the 10 datasets

#1. original dataset
original_set_train = joblib.load(processed_dataset_dir + 'original_set_train' + '.gz')
original_set_test = joblib.load(processed_dataset_dir + 'original_set_test' + '.gz')
original_labels_train = joblib.load(processed_dataset_dir + 'original_labels_train' + '.gz')
original_labels_test = joblib.load(processed_dataset_dir + 'original_labels_test' + '.gz')

#2. ros_boruta dataset
ros_boruta_set_train = joblib.load(processed_dataset_dir + 'ros_boruta_set_train' + '.gz')
ros_boruta_set_test = joblib.load(processed_dataset_dir + 'ros_boruta_set_test' + '.gz')
ros_boruta_labels_train = joblib.load(processed_dataset_dir + 'ros_boruta_labels_train' + '.gz')
ros_boruta_labels_test = joblib.load(processed_dataset_dir + 'ros_boruta_labels_test' + '.gz')

#3. ros_l1 dataset
ros_l1_set_train = joblib.load(processed_dataset_dir + 'ros_l1_set_train' + '.gz')
ros_l1_set_test = joblib.load(processed_dataset_dir + 'ros_l1_set_test' + '.gz')
ros_l1_labels_train = joblib.load(processed_dataset_dir + 'ros_l1_labels_train' + '.gz')
ros_l1_labels_test = joblib.load(processed_dataset_dir + 'ros_l1_labels_test' + '.gz')

#4. ros_tr dataset
ros_tr_set_train = joblib.load(processed_dataset_dir + 'ros_tr_set_train' + '.gz')
ros_tr_set_test = joblib.load(processed_dataset_dir + 'ros_tr_set_test' + '.gz')
ros_tr_labels_train = joblib.load(processed_dataset_dir + 'ros_tr_labels_train' + '.gz')
ros_tr_labels_test = joblib.load(processed_dataset_dir + 'ros_tr_labels_test' + '.gz')

#5. renn_boruta dataset
renn_boruta_set_train = joblib.load(processed_dataset_dir + 'renn_boruta_set_train' + '.gz')
renn_boruta_set_test = joblib.load(processed_dataset_dir + 'renn_boruta_set_test' + '.gz')
renn_boruta_labels_train = joblib.load(processed_dataset_dir + 'renn_boruta_labels_train' + '.gz')
renn_boruta_labels_test = joblib.load(processed_dataset_dir + 'renn_boruta_labels_test' + '.gz')

#6. renn_l1 dataset
renn_l1_set_train = joblib.load(processed_dataset_dir + 'renn_l1_set_train' + '.gz')
renn_l1_set_test = joblib.load(processed_dataset_dir + 'renn_l1_set_test' + '.gz')
renn_l1_labels_train = joblib.load(processed_dataset_dir + 'renn_l1_labels_train' + '.gz')
renn_l1_labels_test = joblib.load(processed_dataset_dir + 'renn_l1_labels_test' + '.gz')

#7. renn_tr dataset
renn_tr_set_train = joblib.load(processed_dataset_dir + 'renn_tr_set_train' + '.gz')
renn_tr_set_test = joblib.load(processed_dataset_dir + 'renn_tr_set_test' + '.gz')
renn_tr_labels_train = joblib.load(processed_dataset_dir + 'renn_tr_labels_train' + '.gz')
renn_tr_labels_test = joblib.load(processed_dataset_dir + 'renn_tr_labels_test' + '.gz')

#8. smote_boruta dataset
smote_boruta_set_train = joblib.load(processed_dataset_dir + 'smote_boruta_set_train' + '.gz')
smote_boruta_set_test = joblib.load(processed_dataset_dir + 'smote_boruta_set_test' + '.gz')
smote_boruta_labels_train = joblib.load(processed_dataset_dir + 'smote_boruta_labels_train' + '.gz')
smote_boruta_labels_test = joblib.load(processed_dataset_dir + 'smote_boruta_labels_test' + '.gz')

#9. smote_l1 dataset
smote_l1_set_train = joblib.load(processed_dataset_dir + 'smote_l1_set_train' + '.gz')
smote_l1_set_test = joblib.load(processed_dataset_dir + 'smote_l1_set_test' + '.gz')
smote_l1_labels_train = joblib.load(processed_dataset_dir + 'smote_l1_labels_train' + '.gz')
smote_l1_labels_test = joblib.load(processed_dataset_dir + 'smote_l1_labels_test' + '.gz')

#10. smote_tr dataset
smote_tr_set_train = joblib.load(processed_dataset_dir + 'smote_tr_set_train' + '.gz')
smote_tr_set_test = joblib.load(processed_dataset_dir + 'smote_tr_set_test' + '.gz')
smote_tr_labels_train = joblib.load(processed_dataset_dir + 'smote_tr_labels_train' + '.gz')
smote_tr_labels_test = joblib.load(processed_dataset_dir + 'smote_tr_labels_test' + '.gz')

train_set_list = [original_set_train,ros_boruta_set_train,ros_l1_set_train,ros_tr_set_train, \
                 renn_boruta_set_train,renn_l1_set_train,renn_tr_set_train,smote_boruta_set_train,\
                 smote_l1_set_train,smote_tr_set_train]

dataset_name_list = ["original","ros_boruta","ros_l1","ros_tr","renn_boruta", \
                     "renn_l1","renn_tr","smote_boruta","smote_l1","smote_tr"]

train_labels_list = [original_labels_train,ros_boruta_labels_train,ros_l1_labels_train, \
                    ros_tr_labels_train,renn_boruta_labels_train,renn_l1_labels_train \
                    ,renn_tr_labels_train,smote_boruta_labels_train,smote_l1_labels_train,smote_tr_labels_train]
test_set_list = [original_set_test,ros_boruta_set_test,ros_l1_set_test,ros_tr_set_test,renn_boruta_set_test, \
                renn_l1_set_test,renn_tr_set_test,smote_boruta_set_test,smote_l1_set_test,smote_tr_set_test]
test_labels_list = [original_labels_test,ros_boruta_labels_test,ros_l1_labels_test,ros_tr_labels_test, \
                  renn_boruta_labels_test,renn_l1_labels_test,renn_tr_labels_test,smote_boruta_labels_test, \
                  smote_l1_labels_test,smote_tr_labels_test]


for i in range(len(train_set_list)):
    train_set_list[i] = csr_matrix.toarray(train_set_list[i])
    test_set_list[i] = csr_matrix.toarray(test_set_list[i])

classifier_list = [ \
    DecisionTreeClassifier(), \
    LinearSVC(random_state=0, tol=1e-5), \
    neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance'), \
    GaussianNB(), \
    SkopeRules(), \
    RandomForestClassifier(n_estimators=100, random_state=0), \
    AdaBoostClassifier(n_estimators=100), \
    VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), \
                            ('knn', neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')), \
                            ('gnb', GaussianNB())], voting='hard')\
]

model_name_list = ["decision_tree", "linear_svc", "knn", "gaussianNB", "skope_rule", "random_forest", "boosting", "hybrid"]

model_name = 'skopeRules'
f = open(model_name + "_result.txt","w+")

def model(i):
    print("running dataset:",i)
    start_time = time.time()
    f = open(model_name + "_result.txt","a")
    rule_clf = SkopeRules(max_depth_duplication=None,
                    n_estimators=30,
                    precision_min=0.2,
                    recall_min=0.01)

    #apply ftwo score to evaluate models
    ftwo_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)
    #train the model against the original dataset
    cross_val_score(rule_clf, train_set_list[i], train_labels_list[i], scoring = ftwo_scorer, cv=10)
    end_time = time.time()
    train_time_duration = end_time - start_time
    start_time = time.time()
    prediction = cross_val_predict(rule_clf, test_set_list[i], test_labels_list[i], cv=10)
    end_time = time.time()
    test_time_duration = end_time - start_time

    fbeta_score = metrics.fbeta_score(test_labels_list[i], prediction, average='macro', beta=2)
    confusion_matrix_ = confusion_matrix(test_labels_list[i], prediction)
    accuracy_score_ = accuracy_score(test_labels_list[i], prediction)
    precision_score_ = precision_score(test_labels_list[i], prediction)
    recall_score_ = recall_score(test_labels_list[i], prediction)
    roc_auc_score_ = roc_auc_score(test_labels_list[i], prediction)
    log_loss_ = log_loss(test_labels_list[i], prediction)

    
    f.write("dataset:"+dataset_name_list[i]+";\r\n")
    f.write("fbeta_score:"+str(fbeta_score)+";\r\n")
    f.write("confusion_matrix_:"+str(confusion_matrix_)+";\r\n")
    f.write("accuracy_score_:"+str(accuracy_score_)+";\r\n")
    f.write("precision_score_:"+str(precision_score_)+";\r\n")
    f.write("recall_score_:"+str(recall_score_)+";\r\n")
    f.write("auc_score_:"+str(roc_auc_score_)+";\r\n")
    f.write("log_loss_:"+str(log_loss_)+";\r\n")
    f.write("test_time_duration:"+str(test_time_duration)+";\r\n")
    f.write("train_time_duration:"+str(train_time_duration)+"s,\r\n\r\n")
    # fpr, tpr, _ = roc_curve(test_labels_list[i], prediction)
    f.close()

@profile
def model_train_with_memery_usage():
    # for i in range(10):
    #     model(i)
    model(0)

model_train_with_memery_usage()