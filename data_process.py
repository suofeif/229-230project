#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors


# load data from csv data files
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

def load_word2vec_model():
    #wv_from_text = KeyedVectors.load_word2vec_format("../tencentVec/Tencent_AILab_ChineseEmbedding.txt", binary = False)

    return wv_from_text

def train_w2v(sentences):



'''print("import ended...")

train_data_path = os.path.abspath('..') + "/data/trainingset.csv"
validate_data_path = os.path.abspath('..') + "/data/validationset.csv"
test_data_path = os.path.abspath('..') + "/data/testa.csv"

print("read csv started...")
data = pd.read_csv(train_data_path)
data_validation = pd.read_csv(validate_data_path)
data_test = pd.read_csv(test_data_path)
print("training.shape:",data.shape)
print("data_validation.shape:",data_validation.shape)
print("data_test.shape:",data_test.shape)
print("read csv ended...")

print(data.head(n=1))
data.shape
# print some line to have a look
for index, row in data.iterrows():
    id_=row['id']
    content=row['content']
    label=row[2:]
    print("id_:",id_)
    print("content:",content)
    print("label:",label)
    print("===============================================================")
    if index==5:
        break

def get_sentiment_analysis_labels(row):
  # 1)location
    location_traffic_convenience = row['location_traffic_convenience']
    location_distance_from_business_district= row['location_distance_from_business_district']
    location_easy_to_find  = row['location_easy_to_find']
 # 2)service
    service_wait_time  = row['service_wait_time']
    service_waiters_attitude = row['service_waiters_attitude']
    service_parking_convenience = row['service_parking_convenience']
    service_serving_speed  = row['service_serving_speed']
 # 3)price
    price_level = row['price_level']
    price_cost_effective = row['price_cost_effective']
    price_discount  = row['price_discount']
 # 4)environment
    environment_decoration  = row['environment_decoration']
    environment_noise   = row['environment_noise']
    environment_space   = row['environment_space']
    environment_cleaness     = row['environment_cleaness']
    # 5)dish
    dish_portion   = row['dish_portion']
    dish_taste =row['dish_taste']
    dish_look  = row['dish_look']
    dish_recommendation = row['dish_recommendation']
    # 6)other
    others_overall_experience  = row['others_overall_experience']
    others_willing_to_consume_again   = row['others_willing_to_consume_again']

    label_list=[]
    label_list=[location_traffic_convenience,location_distance_from_business_district,location_easy_to_find, # location
        service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed, # service
        price_level,price_cost_effective,price_discount, # price
        environment_decoration,environment_noise,environment_space,environment_cleaness, # environment
        dish_portion,dish_taste,dish_look,dish_recommendation, # dish
        others_overall_experience,others_willing_to_consume_again] # other
    label_list=[str(i)+"_"+str(label_list[i]) for i  in range(len(label_list))]

    return label_list

def convet_to_one_hot(label_list,num_classes=80):
    new_label_list=[0 for i in range(num_classes)]
    for label in label_list:
        new_label_list[label]=1
    return new_label_list

    label_list=[0, 4, 8, 12, 19, 20, 24, 28, 32, 39, 43, 47, 51, 55, 59, 62, 64, 68, 75, 76]
    label_list_one_hot=convet_to_one_hot(label_list)
    print("label_list_one_hot:",label_list_one_hot)

    # print first row
for index, row in data.iterrows():
    id_=row['id']
    content=[x for x in jieba.lcut(row['content']) if x.strip() and x!="\""]
    sentiment_label_list=get_sentiment_analysis_labels(row)
    print("id_:",id_)
    print("content:",content)
    print("sentiment_label_list:",sentiment_label_list,";length:",len(sentiment_label_list))
    if index==0:
        break


# sample a some percentage of data(set frac to small value when you are in test model) and create vocabulary by get most requency words
vocab_size=70000

print("df.shape:",data.shape)

PAD_ID = 0
UNK_ID=1
CLS_ID=2
MASK_ID=3
_PAD="_PAD"
_UNK="UNK"
_CLS="CLS"
_MASK="MASK"

data_traininig_small=data.sample(frac=1.0)
data_validation_small=data_validation.sample(frac=1.0)
data_test_small=data_test #[0:1000]
data_big=pd.concat([data_traininig_small,data_validation_small,data_test_small])#(data_traininig_small, data_test_small, on='key')
print("data_big:",data_big.shape)

print("data_traininig_small.shape:",data_traininig_small.shape)
print("data_validation_small.shape:",data_validation_small.shape)
print("data_test_small.shape:",data_test_small.shape)

num_example,_=data_traininig_small.shape
print("num_example:",num_example)
total_length=0
c_inputs=Counter()
count_index=0
for index, row in data_big.iterrows():
    #id_=row['id']
    input_list=[x for x in jieba.lcut(row['content']) if x.strip() and x!="\""]
    total_length+=len(input_list)
    c_inputs.update(input_list)
    count_index=count_index+1
    if count_index%5000==0:
        print("count.create vocabulary of words:",count_index)

vocab_list=c_inputs.most_common(vocab_size-4)
#print("vocab_list:",vocab_list)
vocab_word2index={}
vocab_word2index[_PAD]=PAD_ID
vocab_word2index[_UNK]=UNK_ID
vocab_word2index[_CLS]=CLS_ID
vocab_word2index[_MASK]=MASK_ID

count_index=0
for i,tuplee in enumerate(vocab_list):
    word,freq=tuplee
    vocab_word2index[word]=i+4
#print("vocab_word2index:",vocab_word2index)


# compute length distribution of inputs and set max_sequence_length;
# choose a mininum value of max_sequence_length that let 90% of inputs' s length is less of it.
total_length=0
input_length_list=[50,100,150,200,250,300,350,400,500,600,700,1000]
input_length_dict={x:0 for x in input_length_list }
data_big_part=data_big.sample(frac=0.25)

num_example_for_length,_=data_big_part.shape
print("num_example_for_length:",num_example_for_length)
for index, row in data_big_part.iterrows(): # data_traininig_small
    id_=row['id']
    input_list=[x for x in jieba.lcut(row['content']) if x.strip() and x!="\""]
    length_input=len(input_list)
    fixed_len=1000
    for length in input_length_list:
        if length_input<length:
            fixed_len=length
            break
    input_length_dict[fixed_len]=input_length_dict[fixed_len]+1
    total_length+=length_input
    c_inputs.update(input_list)

avg_length=float(total_length) /float(num_example_for_length)
print("avg_length:",avg_length)
print("input_length_dict:",input_length_dict)

input_length_dict_percentage={}
for k,v in input_length_dict.items():
    v=v/num_example_for_length
    input_length_dict_percentage[k]=v
print("input_length_dict_percentage:",input_length_dict_percentage)

# conclusion: most length is between 150-250. average length is 216. if we use max length 350, then we can cover about:  91% of inputs.
# choose a mininum value of max_sequence_length that let 90% of inputs' s length is less of it.
max_sequence_length=0
accumulate_percentage=0
for fixed_length,percentage in input_length_dict_percentage.items():
    accumulate_percentage+=percentage
    if accumulate_percentage>0.9:
        max_sequence_length=fixed_length
        print("max_sequence_length:",max_sequence_length)
        break

# create dict of label to index
value_list=[-2,-1,0,1]
num_aspect=20
label2index={}
count_label=0
for i in range(num_aspect):
    for value in value_list:
        label2index[str(i)+"_"+str(value)]=count_label
        count_label+=1
print("label2index:",label2index)

# generate training data(X,y) by using vocabulary of words, padding sequence
def process_data(data,data_type='train_valid'):
    X,Y=[],[]
    print(data.shape)
    count=0
    for index, row in data.iterrows():
            input_list=[x for x in jieba.lcut(row['content']) if x.strip() and x!="\""]
            x=[vocab_word2index.get(x,UNK_ID) for x in input_list]
            X.append(x)
            if data_type=='train_valid':
                sentiment_label_list=get_sentiment_analysis_labels(row)
                y_dense=[label2index[label] for label in sentiment_label_list]
                y=convet_to_one_hot(y_dense)
                Y.append(y)
                count=count+1
                if count%10000==0:
                    if data_type=='train_valid':
                        print(count,",sentiment_label_list:",sentiment_label_list)
                        print(count,",y_dense:",y_dense)
                        print(count,",y:",y)
                    print("===================================================")'''
            '''if count%10000==0:
                print(count,",input_list:",input_list)
                print(count,",x:",x)
    return X,Y

print("process_data.started...")
X_train,Y_train=process_data(data_traininig_small)
print("process_data.train.ended.")
X_valid,Y_valid=process_data(data_validation_small)
print("process_data.valid.ended.")
X_test,_=process_data(data_test_small,data_type='test')
print("process_data.ended...")

print("X_train:",np.array(X_train).shape,";Y_train:",np.array(Y_train).shape)
print("X_valid:",np.array(X_valid).shape,";Y_valid:",np.array(Y_valid).shape)
print("X_test:",np.array(X_test).shape)
print("X_train[0]:",X_train[0])

# pad input x to fixed length
import tensorflow as tf
pad_sequence = tf.keras.preprocessing.sequence.pad_sequences
print("max_sequence_length:",max_sequence_length)
X_train = pad_sequence(np.array(X_train),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)
X_valid = pad_sequence(np.array(X_valid),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)
X_test = pad_sequence(np.array(X_test),maxlen=max_sequence_length,padding='pre',truncating='pre',value = 0)

X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_valid=np.array(X_valid)
Y_valid=np.array(Y_valid)
X_test=np.array(X_test)
print("X_train:",X_train.shape,";Y_train:",Y_train.shape)
print("X_valid:",X_valid.shape,";Y_valid:",Y_valid.shape)
print("X_test:",X_test.shape)
print("X_train[0]:",X_train[0])

# save train/validation/test set into cache file
cache_path=os.path.abspath('..') + "/output/pre_processed_vocab_cache.pik" #'data/train_valid_test_vocab_cache.pik'
#if not os.path.exists(cache_path):
with open(cache_path, 'ab') as data_f:
    print("going to save cache file of vocab of words and labels")
    pickle.dump((X_train,Y_train,X_valid,Y_valid,X_test,vocab_word2index,label2index), data_f)
    print("write train/validation/test set into cache file completed.")'''
