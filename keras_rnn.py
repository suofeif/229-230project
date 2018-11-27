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
print("finish loading labels")
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)

#tokenize inputs and train word2vec matrix/load tencent model
seg_train = seg_words(train_doc)
seg_val = seg_words(val_doc)
seg_test = seg_words(test_doc)
print("finish segmenting")

vocab, embedding_model = train_vec(seg_train)
#vocab, embedding_model = train_tencent_model()

# pad documents to a max length of 4 words
max_length = 350
#padded_train_docs = pad_sequences(seg_train, maxlen=max_length, padding='post')
#padded_val_docs = pad_sequences(seg_val, maxlen=max_length, padding='post')
#padded_test_docs = pad_sequences(seg_test, maxlen=max_length, padding='post')
#print(padded_docs[1:3])

# integer encode the documents
word2index = wordToIndex(vocab)
train_indices = sentence_to_indice(seg_train, word2index, max_length, vocab)
val_indices = sentence_to_indice(seg_val, word2index, max_length, vocab)
test_indices = sentence_to_indice(seg_test, word2index, max_length, vocab)

#make embedding_matrix
vocab_len = len(word2index)+1
emb_dim = 300
#emb_dim=200
embedding_matrix = embedding_data(vocab_len, emb_dim, word2index, embedding_model, vocab)

# define the model
n = train_labels.shape[1]
model=dict()
for i in range(n):
	logger.info("start train column" +str(i))

	model = Sequential()
	model.add(Embedding(vocab_len, emb_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
	model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
	#model.add(Flatten())
	model.add(Dense(4, activation='softmax'))
	# compile the model
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
	# summarize the model
	print(model.summary())
	# fit the model
	n = train_labels.shape[1]
	#one_hot y
	y=np.array(convert_to_onehot(train_labels[:, i], 4))
	#print(y)
	model.fit(train_indices, y, epochs=2, verbose=0)

	# evaluate the model
	y=np.array(convert_to_onehot(test_labels[:, i], 4))
	loss, accuracy = model.evaluate(test_indices, y, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

	logger.info("complete train model" + str(i))
	#model[i] = model #store model to dict

logger.info("complete train model")
logger.info("start save model")
model_save_path = config.model_save_path
if not os.path.exists(model_save_path):
	os.makedirs(model_save_path)


logger.info("complete save model")
