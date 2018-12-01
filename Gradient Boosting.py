import numpy as np
import config
import os
import logging
from numpy import array
import numpy as np
import scipy as sp
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import config
import os
import logging
from pre_processing_ref import load_data_from_csv, seg_words, train_vec, train_tencent_model, convert_to_onehot, sentence_to_indice, embedding_data, wordToIndex


# define documents
train=load_data_from_csv(config.train_data_path)
val = load_data_from_csv(config.validate_data_path)
print("finish loading data")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
#test = load_data_from_csv(config.test_data_path)

#val dataset as hold-out val set and test dataset; do cv on train dataset
#m_val = val.shape[0]
#train_doc = train[:, 1]
#val_doc = val[0:7500, 1]
#test_doc = val[7501:m_val-1, 1]
m_val = val.shape[0]
train_doc = train.iloc[:, 1]
val_doc = val.iloc[0:7500:, 1]
test_doc = val.iloc[7501:m_val-1, 1]
print("finish splitting")
print(train_doc.shape)
print(val_doc.shape)
print(test_doc.shape)

# define class labels
#train_labels = array(train[:, 2:])
#val_labels = array(val[0:7500, 2:])
#test_labels = array(val[7501:m_val-1, 2:])
train_labels = array(train.iloc[:, 2:])
val_labels = array(val.iloc[0:7500:, 2:])
test_labels = array(val.iloc[7501:m_val-1:, 2:])
#print("test number"+str(test_labels.shape[0]))
print("finish loading labels")
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)

train_indices = np.load("/Users/EzizDurdyev/Downloads/train_indices.dat")
val_indices = np.load("/Users/EzizDurdyev/Downloads/val_indices.dat")
test_indices = np.load("/Users/EzizDurdyev/Downloads/test_indices.dat")
max_length=train_indices.shape[1]
print("max_length")
print(max_length)

# Grid Search
params = {'learning_rate': [0.05, 0.1],
          'max_depth': [3, 5]
          }

clf = xgb.XGBClassifier()

cv = GridSearchCV(clf, params, cv = 10, scoring = "f1", n_jobs = -1, verbose = 2)

cv.fit(train_indices, train_labels)
predict = cv.predict(val_indices)





print (cv.best_estimator_)
print (cv.best_params_)



best_param = {'learning_rate': cv.best_estimator_.learning_rate,
              'max_depth': cv.best_estimator_.max_depth
              }


dtrain = xgb.DMatrix(np.array(train_indices), label= train_labels)
num_round = 10
bst = xgb.train(best_param, dtrain, num_round)

dtest_x = xgb.DMatrix(np.array(val_indices))
predict = bst.predict(dtest_x)
total_num = val_labels.shape[0]
score = 0
for i,v in enumerate(predict):
    if np.argmax(v) == val_labels[i]:
        score += 1
print ("accuracy is {acc}".format(acc = score/total_num))

