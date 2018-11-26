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

'''def train_tencent_model():
    wv_from_text = KeyedVectors.load_word2vec_format("../tencentVec/Tencent_AILab_ChineseEmbedding.txt", binary = False)


    return wv_from_text

def vec_useTencent():
    train=load_data_from_csv(config.train_data_path)
    val = load_data_from_csv(config.validate_data_path)
    test = load_data_from_csv(config.test_data_path)
    train_debug = train[1:10, 1]
    val_debug = val[1:10, 1]
    test_debug = test[1:10, 1]
    train_seg = seg_words(train_debug)
    val_seg = seg_words(val_debug)
    test_seg = seg_words(test_debug)

    vecModel = load_tencent_model()
    '''

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
    model = Word2Vec(sentences,size=300, window = 5, min_count=5)

    return model

def embedding_data():
    train=load_data_from_csv(config.train_data_path)
    val = load_data_from_csv(config.validate_data_path)
    test = load_data_from_csv(config.test_data_path)
    train_debug = train[1:10, 1]
    val_debug = val[1:10, 1]
    test_debug = test[1:10, 1]
    train_seg = seg_words(train_debug)
    val_seg = seg_words(val_debug)
    test_seg = seg_words(test_debug)
    train_vec = []
    val_vec = []
    test_vec = []

    model = Word2Vec.load("vec_model")



def pad_data()
    import tensorflow as tf
    pad_sequence = tf.keras.preprocessing.sequence.pad_sequences
    print("max_sequence_length:",max_sequence_length)
    X_train = pad_sequence(np.array(X_train),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)
    X_valid = pad_sequence(np.array(X_valid),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)
    X_test = pad_sequence(np.array(X_test),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)





logger.info("start load data")
train_data_df = load_data_from_csv(config.train_data_path)
validate_data_df = load_data_from_csv(config.validate_data_path)

content_train = train_data_df.iloc[:, 1]

logger.info("start seg train data")
content_train = seg_words(content_train)
logger.info("complete seg train data")

columns = train_data_df.columns.values.tolist()

#feature extraction: bag-of-words based
logger.info("start train feature extraction")
model = train_vec(content_train)
model.save("vec_model")
train_vec, val_vec, test_vec = embedding_data()
logger.info("complete train feature extraction models")
logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))
