from numpy import array
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import config
# define documents
train=load_data_from_csv(config.train_data_path)
val = load_data_from_csv(config.validate_data_path)
#test = load_data_from_csv(config.test_data_path)

#val dataset as hold-out val set and test dataset; do cv on train dataset
m_val = val.shape[0]
train_doc = train[:, 1]
val_doc = val[0:7500, 1]
test_doc = val[7501:m_val-1, 1]

# define class labels
train_labels = array(train[:, 2:])
val_labels = array(val[0:7500, 2:])
test_labels = array(val[7501:m_val-1, 2:])
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
model={}
for i in range(n):
	model["model"+ str(i)] = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Flatten())
model.add(Dense(4, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
label_n = train_labels.shape[1]
for col in range(n):
	y = train_labels[:, col]
	model.fit(padded_train_docs, y, epochs=50, verbose=0)

# evaluate the model
for col in range(n):
	y = test_labels[:, col]
	loss, accuracy = model.evaluate(padded_test_docs, y, verbose=0)
	print('Accuracy: %f' % (accuracy*100))
