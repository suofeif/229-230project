import numpy as np
from numpy import array
import tensorflow as tf
import keras
import config
import os
from keras.models import model_from_json
from pre_processing_ref import load_data_from_csv, seg_words, train_vec, train_tencent_model, convert_to_onehot, sentence_to_indice, embedding_data, wordToIndex
# define documents
train=load_data_from_csv(config.train_data_path)
val = load_data_from_csv(config.validate_data_path)
print("finish loading data")

#test = load_data_from_csv(config.test_data_path)

#val dataset as hold-out val set and test dataset; do cv on train dataset
m_val = val.shape[0]
train_doc = train.iloc[:, 1]
val_doc = val.iloc[0:7500, 1]
test_doc = val.iloc[7501:m_val-1, 1]

'''
train_doc = train.iloc[1:10, 1]
val_doc = val.iloc[0:5, 1]
test_doc = val.iloc[6:10, 1]'''

print("finish splitting")
print(train_doc.shape)
print(val_doc.shape)
print(test_doc.shape)


#tokenize inputs and train word2vec matrix/load tencent model
seg_train = seg_words(train_doc)
seg_val = seg_words(val_doc)
seg_test = seg_words(test_doc)
print("finish segmenting")

vocab, embedding_model = train_vec(seg_train)
#vocab, embedding_model = train_tencent_model()

# pad documents to a max length of 4 words
max_length = 350

# integer encode the documents
word2index = wordToIndex(vocab)
train_indices = sentence_to_indice(seg_train, word2index, max_length, vocab)
val_indices = sentence_to_indice(seg_val, word2index, max_length, vocab)
test_indices = sentence_to_indice(seg_test, word2index, max_length, vocab)

#save indices files
train_indices.dump("train_indices.dat")
val_indices.dump("val_indices.dat")
test_indices.dump("test_indices.dat")
print("finish saving indices")

#make embedding_matrix
vocab_len = len(word2index)+1
emb_dim = 300
#emb_dim=200
embedding_matrix = embedding_data(vocab_len, emb_dim, word2index, embedding_model, vocab)
embedding_matrix.dump("embedding_matrix.dat")
print("finish saving embedding_matrix")
#save embedding_matrix
