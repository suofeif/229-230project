import numpy as np
from numpy import array
import tensorflow as tf
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras import optimizers
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import config
import os
import logging
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
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
train_doc = train.iloc[1:10, 1]
val_doc = val.iloc[0:5, 1]
test_doc = val.iloc[6:10, 1]
print("finish splitting")
print(train_doc.shape)
print(val_doc.shape)
print(test_doc.shape)

# define class labels
#train_labels = array(train[:, 2:])
#val_labels = array(val[0:7500, 2:])
#test_labels = array(val[7501:m_val-1, 2:])
train_labels = array(train.iloc[1:10, 2:])
val_labels = array(val.iloc[0:5, 2:])
test_labels = array(val.iloc[6:10, 2:])
#print("test number"+str(test_labels.shape[0]))
print("finish loading labels")
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)

train_indices = np.load("train_indices.dat")
val_indices = np.load("val_indices.dat")
test_indices = np.load("test_indices.dat")
max_length=train_indices.shape[1]
print("max_length")
print(max_length)

embedding_matrix=np.load("embedding_matrix.dat")
vocab_len = embedding_matrix.shape[0]
emb_dim = embedding_matrix.shape[1]
print("vocab_len")
print(vocab_len)
print("emb_dim")
print(emb_dim)

# define the model
DROPOUT=0.3
NUM_HID=16
REC_DROP=0.3
n = train_labels.shape[1]
#model=dict()
for i in range(n):
	logger.info("start train column" +str(i))

	model = Sequential()
	#model.add(Input(shape=(max_length,), dtype='int32'))
	model.add(Embedding(vocab_len, emb_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
	model.add(LSTM(NUM_HID, dropout=DROPOUT, recurrent_dropout=REC_DROP,return_sequences=True))
	model.add(LSTM(NUM_HID, dropout=DROPOUT, recurrent_dropout=REC_DROP,return_sequences=False))
	#model.add(Flatten())
	model.add(Dense(4))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	# compile the model
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
	# summarize the model
	print(model.summary())
	# fit the model
	n = train_labels.shape[1]
	#one_hot y
	y_train=np.array(convert_to_onehot(train_labels[:, i], 4))
	y_val=np.array(convert_to_onehot(val_labels[:, i], 4))
	#print(y)
	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	model.fit(train_indices, y_train, validation_data=(val_indices, y_val), epochs=2, verbose=0, callbacks = [early_stopping])

	# evaluate the model
	y_test=np.array(convert_to_onehot(test_labels[:, i], 4))
	loss, accuracy = model.evaluate(test_indices, y_test, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

	logger.info("complete train model" + str(i))
	#model.save(str(i) + 'th model.h5')
	print("complete saving model")
	del model
	#model[i] = model #store model to dict

logger.info("complete train model")
