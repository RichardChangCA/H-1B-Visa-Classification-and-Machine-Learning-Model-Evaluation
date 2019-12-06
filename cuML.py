# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import time
import joblib
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier as RFC
from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.model_selection import train_test_split

# Using NumPy for number of class detection,install CuPy for faster processing.

# Where to save the figures
processed_dataset_dir = "deleted_dataset/"

#1. original dataset
original_set_train = joblib.load(processed_dataset_dir + 'original_set_train' + '.gz')
original_set_test = joblib.load(processed_dataset_dir + 'original_set_test' + '.gz')
original_labels_train = joblib.load(processed_dataset_dir + 'original_labels_train' + '.gz')
original_labels_test = joblib.load(processed_dataset_dir + 'original_labels_test' + '.gz')

# dataset = joblib.load(processed_dataset_dir + 'raw_set' + '.gz')
# dataset_label = joblib.load(processed_dataset_dir + 'raw_labels' + '.gz')

# dataset = csr_matrix.toarray(dataset)

# dataset_label = np.array(dataset_label,dtype = np.int32)

# original_set_train, original_set_test, original_labels_train, original_labels_test = train_test_split(dataset,dataset_label, test_size=0.2, random_state=42)



original_labels_train = np.array(original_labels_train,dtype = np.int32)
original_labels_test = np.array(original_labels_test,dtype = np.int32)
original_set_train = csr_matrix.toarray(original_set_train)
original_set_test = csr_matrix.toarray(original_set_test)

training_start_time = time.time()
cuml_model = cuRFC(max_features=1.0,n_estimators=40)
cuml_model.fit(original_set_train,original_labels_train)
training_end_time = time.time()
print("training_duration_with_GPU:",training_end_time - training_start_time)

testing_start_time = time.time()
cuml_predict = cuml_model.predict(original_set_test)
testing_end_time = time.time()

print("testing_duration_with_GPU:",testing_end_time - testing_start_time)


training_start_time = time.time()
ml_model = RFC(max_features=1.0,n_estimators=40)
ml_model.fit(original_set_train,original_labels_train)
training_end_time = time.time()
print("training_duration:",training_end_time - training_start_time)

testing_start_time = time.time()
ml_predict = ml_model.predict(original_set_test)
testing_end_time = time.time()

print("testing_duration:",testing_end_time - testing_start_time)