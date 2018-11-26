from numpy import array
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import config
from pre_processing_ref import seg_words, train_vec
# define documents
train=load_data_from_csv(config.train_data_path)
val = load_data_from_csv(config.validate_data_path)
print("finish loading data")
#test = load_data_from_csv(config.test_data_path)

#val dataset as hold-out val set and test dataset; do cv on train dataset
m_val = val.shape[0]
train_doc = train[:, 1]
val_doc = val[0:7500, 1]
test_doc = val[7501:m_val-1, 1]
print("finish splitting")
print(train_doc.shape)
print(val_doc.shape)
print(test_doc.shape)

# define class labels
train_labels = array(train[:, 2:])
val_labels = array(val[0:7500, 2:])
test_labels = array(val[7501:m_val-1, 2:])
print("finish loading labels")
print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)

#tokenize inputs and train word2vec matrix
seg_train = seg_words(train_doc)
seg_val = seg_words(val_doc)
seg_test = seg_words(test_doc)
print("finish segmenting")

embedding_mat = train_vec(seg_train)

# integer encode the documents
vocab_size = 70000
train_encoded_docs = [one_hot(d, vocab_size) for d in train_doc]
val_encoded_docs = [one_hot(d, vocab_size) for d in val_doc]
test_encoded_docs = [one_hot(d, vocab_size) for d in test_doc]
print(encoded_docs[3:5])
# pad documents to a max length of 4 words
max_length = 350
padded_train_docs = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post')
padded_val_docs = pad_sequences(val_encoded_docs, maxlen=max_length, padding='post')
padded_test_docs = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
print(padded_docs[3:5])
# define the model
n = train_labels.shape[1]
model=dict()
for i in range(n):
	logger.info("start train column" +str(i))

	model = Sequential()
	model.add(Embedding(vocab_size, 300, input_length=max_length, weights=[embedding_matrix], trainable=False))
	model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
	#model.add(Flatten())
	model.add(Dense(4, activation='softmax'))
	# compile the model
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
	# summarize the model
	print(model.summary())
	# fit the model
	n = train_labels.shape[1]
	for col in range(n):
		y = train_labels[:, col]
		model.fit(padded_train_docs, y, epochs=50, verbose=0)

	# evaluate the model
	for col in range(n):
		y = test_labels[:, col]
		loss, accuracy = model.evaluate(padded_test_docs, y, verbose=0)
		print('Accuracy: %f' % (accuracy*100))


	logger.info("complete train model" + str(i))
	model[i] = model #store model to dict

logger.info("complete train model")
