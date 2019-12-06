# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Orange CN2 rule-based Anaconda environment

# Common imports
import numpy as np
import os
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
import Orange

# Where to save the figures
processed_dataset_dir = "deleted_dataset"

#load the 10 datasets
#1. original dataset
train_set = joblib.load(os.path.join(processed_dataset_dir,'original_set_train.gz'))
train_labels = joblib.load(os.path.join(processed_dataset_dir,'original_labels_train.gz'))
# #2. oversampling dataset with boruta feature selection technique
# ros_train_set_boruta = joblib.load(os.path.join(processed_dataset_dir,'ros_train_set_boruta.gz'))
# ros_train_labels_boruta = joblib.load(os.path.join(processed_dataset_dir,'ros_train_labels_boruta.gz'))
# #3. oversampling dataset with L1-based feature selection technique
# ros_train_set_l1 = joblib.load(os.path.join(processed_dataset_dir,'ros_train_set_l1.gz'))
# ros_train_labels_l1 = joblib.load(os.path.join(processed_dataset_dir,'ros_train_labels_l1.gz'))
# #4. oversampling dataset with tree-based feature selection technique
# ros_train_set_tr = joblib.load(os.path.join(processed_dataset_dir,'ros_train_set_tr.gz'))
# ros_train_labels_tr = joblib.load(os.path.join(processed_dataset_dir,'ros_train_labels_tr.gz'))
# #5. under-sampling dataset with boruta feature selection technique
# renn_train_set_boruta = joblib.load(os.path.join(processed_dataset_dir,'renn_train_set_boruta.gz'))
# renn_train_labels_boruta = joblib.load(os.path.join(processed_dataset_dir,'renn_train_labels_boruta.gz'))
# #6. under-sampling dataset with L1-based feature selection technique
# renn_train_set_l1 = joblib.load(os.path.join(processed_dataset_dir,'renn_train_set_l1.gz'))
# renn_train_labels_l1 = joblib.load(os.path.join(processed_dataset_dir,'renn_train_labels_l1.gz'))
# #7. under-sampling dataset with tree-based feature selection technique
# renn_train_set_tr = joblib.load(os.path.join(processed_dataset_dir,'renn_train_set_tr.gz'))
# renn_train_labels_tr = joblib.load(os.path.join(processed_dataset_dir,'renn_train_labels_tr.gz'))
# #8. balanced sampling dataset with boruta feature selection technique
# smote_train_set_boruta = joblib.load(os.path.join(processed_dataset_dir,'smote_train_set_boruta.gz'))
# smote_train_labels_boruta = joblib.load(os.path.join(processed_dataset_dir,'smote_train_labels_boruta.gz'))
# #9. balanced sampling dataset with L1-based feature selection technique
# smote_train_set_l1 = joblib.load(os.path.join(processed_dataset_dir,'smote_train_set_l1.gz'))
# smote_train_labels_l1 = joblib.load(os.path.join(processed_dataset_dir,'smote_train_labels_l1.gz'))
# #10. balanced sampling dataset with tree-based feature selection technique
# smote_train_set_tr = joblib.load(os.path.join(processed_dataset_dir,'smote_train_set_tr.gz'))
# smote_train_labels_tr = joblib.load(os.path.join(processed_dataset_dir,'smote_train_labels_tr.gz'))

model_name = 'CN2 rule-based'


train_set = csr_matrix.toarray(train_set)

cn2_learner = Orange.classification.rules.CN2Learner()
domain = Orange.data.Domain.from_numpy(train_set, train_labels)

data = Orange.data.Table.from_numpy(domain=domain,X=train_set,Y=train_labels)

scores = Orange.evaluation.CrossValidation(data,[cn2_learner],k=10)

accuracy = Orange.evaluation.scoring.CA(scores)

precesion = Orange.evaluation.scoring.precision(scores)

recall = Orange.evaluation.scoring.Recall(scores)

print("precision of " + model_name + " model on original dataset is + %f" % precision)
print("recall of " + model_name + " model on original dataset is + %f" % recall)
print("accuracy of " + model_name + " model on original dataset is + %f" % accuracy)

