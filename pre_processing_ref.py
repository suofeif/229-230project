#refer: https://github.com/brightmart/sentiment_analysis_fine_grain/blob/master/preprocess_word.ipynb

import random
random.seed = 16
import pandas as pd
#from gensim.models.word2vec import Word2Vec
import jieba
from collections import Counter
import numpy as np
import os
import pickle
import config
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import logging

def train_tencent_model():
    wv_from_text = KeyedVectors.load_word2vec_format("../tencentVec/Tencent_AILab_ChineseEmbedding.txt", binary = False)
    vocab = wv_from_text.wv.vocab

    return vocab, wv_from_text



def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df


# text tokenization
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs

def train_vec(sentences):
    model = Word2Vec(sentences,size=300, window = 5, min_count=5, iter = 100)
    save_path = config.embedding_model_save_path
    path = save_path + "word2vec.model"
    if not os.path.exists(path):
    	os.makedirs(path)
    model.save(path)
    wv = KeyedVectors.load("model.wv", mmap='r')

    vocab = model.wv.vocab
    return vocab, wv

def convert_to_onehot(labels, class_num):
    one_hot_mat = []
    for j in range(len(labels)):
        label = labels[j]
        new_labels = [0 for i in range(class_num)]
        for label in labels:
            new_labels[label] = 1
        one_hot_mat.append(new_labels) # list concatenation
    return one_hot_mat

def sentence_to_indice(lists, word2index, max_len):
    X = np.array(lists)
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence = lists[i]
        for j in range(max_len):
            word = sentence[j]
            k = word2index[word]
            X_indices[i, j] = k
    return X_indices

def embedding_data(vocab_len, emb_dim, vocab, embedding_model):
    self.vocab_len = vocab_len
    self.emb_dim = emb_dim

    emb_matrix = np.zeros((vocab_len, emb_dim))
    index=0
    for word in vocab:
        emb_matrix[index, :] = embedding_model[word]
        index = index+1

    return emb_matrix
