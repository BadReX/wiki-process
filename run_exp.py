# coding: utf-8
# Run some experiments using the neural bag-of-words (or deep averaging network)

# basic imports
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import math 
import sys
import os

# set env backend to theano
print('Importing keras and sklearn ...')
os.environ['KERAS_BACKEND']='theano'

# Keras imports
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Activation
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import Embedding, Merge, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adagrad, Adam
from keras.layers import Layer
from keras.callbacks import EarlyStopping

# other imports
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

print('Reading data from disk ...')
wikidocs = pd.read_csv('/home/babdulla/DNN/wiki-data/wikidocs.csv', sep=',')
print('Dimensions of the data frame (r, c):', wikidocs.shape)


print('Prepare data lists ...')
docs = []
labels = []

for idx in range(wikidocs.words.shape[0]): # 
    doc_str = wikidocs.words[idx]
      
    # this condition was added because 2 rows were NaN for unknown reason!
    if isinstance(doc_str, str):
        docs.append(doc_str)
        labels.append(wikidocs.clusters[idx])

print('Counting the number of unique words in the data ...')
vocab = set([w for d in docs for w in d.split()])       
print('Number of unique words (vocab) is ',  len(vocab))

# save some memory space by deleting this var
del wikidocs

print('Tokenizing text in the data and make sequences (this might take some time) ...')
tokenizer = Tokenizer(split=" ", filters="")
tokenizer.fit_on_texts(docs)
docs_seqs = tokenizer.texts_to_sequences(docs)

print('Checking if the word_index of tokenizer object has the same words as the in the docs ...')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# test cases 
assert len(vocab) == len(word_index), "word index length mismatch."
assert max(word_index.values()) == len(vocab), "max word index != vocab size"
print('Vocab tests passed! No problems.')

print('Binarizing and transforming the labels with MultiLabelBinarizer ...')
labels = [l.split() for l in labels]

# vectorize labels in the output
mlb = MultiLabelBinarizer()
bin_labels = mlb.fit_transform(labels)
print('Shape of label tensor:', bin_labels.shape)


# test case
assert bin_labels.shape[0] == len(docs_seqs), "input/output data mismatch."


# read pre-trained word embeddings 
print('Read word embeddings from disk ...')
w2v_DIR = "/home/babdulla/DNN/wiki-processed/word-vectors.400d"

embeddings_index = {}

with open(w2v_DIR, "r") as eFile:
    for line in eFile:
        values = line.split()
        word = values[0]
        if word in vocab:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

print('Total %s word vectors are in the embedding matrix.' % len(embeddings_index))

# make embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, 400))

print('Initializing emnbedding matrix ...')
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector
        
print('Shape of the embedding matrix:', embedding_matrix.shape) 

# make averaged context vectors
# TODO: extract only the vectors that are in the vocabulary 
print('Generating context vectors (this might take some time) ...')
context_vectors = np.zeros((len(docs_seqs), 400))

for i in range(len(docs_seqs)):
    seq = docs_seqs[i]
    context_vectors[i] = np.sum([embedding_matrix[w] for w in seq],  axis=0)
    context_vectors[i] /= len(seq)

print('Shape of the contexts tensor:', context_vectors.shape)  

# make train, dev, and test splits
print('Making train, dev, and test splits ...', context_vectors.shape)  
DEV_SPLIT  = 0.1
TEST_SPLIT = 0.1

indices = np.arange(context_vectors.shape[0])
np.random.shuffle(indices)
context_vectors = context_vectors[indices]
bin_labels = bin_labels[indices]

ndev_samples  = int(DEV_SPLIT * context_vectors.shape[0])
ntest_samples = int(TEST_SPLIT * context_vectors.shape[0])

dev_idx = -ndev_samples - ntest_samples

x_train = context_vectors[:dev_idx]
y_train = bin_labels[:dev_idx]

x_dev = context_vectors[dev_idx:-ndev_samples]
y_dev = bin_labels[dev_idx:-ndev_samples]

x_test = context_vectors[-ndev_samples:]
y_test = bin_labels[-ndev_samples:]

# save some space by del some var
del tokenizer
del context_vectors
del docs_seqs

# some stats about the data 
print('Train split:', x_train.shape[0])
print('Dev split:', x_dev.shape[0])
print('Test split:', x_test.shape[0])
 
# build the model
print('Building a neural model ...') 
dropout_rate = 0.2
layers = [400, 800]
batch_normalised = True

model = Sequential()

for i in range(len(layers)):
    if i == 0:
        # input layer
        model.add(Dense(layers[i], input_shape=(400,)))
    else:
        # hidden layers
        model.add(Dense(layers[i]))
        
    model.add(Activation('relu'))
    if batch_normalised:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

# output layer
model.add(Dense(bin_labels.shape[1]))
if batch_normalised:
        model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
model.add(Activation('sigmoid'))

print('Model summary.:')  
model.summary()

adam = Adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


print('Training a neural model ...') 
earlystop = EarlyStopping(monitor='val_acc', 
				min_delta=0.0001, 
				patience=5, 
				verbose=1, 
				mode='auto')

callbacks_list = [earlystop]

model_info = model.fit(x_train,
                       y_train, 
                       batch_size=124, 
                       callbacks=callbacks_list, 
                       epochs=20, 
                       validation_data=(x_dev,y_dev), 
                       verbose=1)

# use the model to make preditions
print('Evaluating the model ...') 
print("Using the neural model to make predictions ...")
nn_preds = model.predict(x_test)

idx_set = [6000, 4127, 49, 59, 50058, 777, 74854, 11155]

fig, axs = plt.subplots(len(idx_set),1,figsize=(8,14))

for i in range(len(idx_set)):
    axs[i].plot(y_test[idx_set[i]], color='c', alpha=1, linewidth=1)
    axs[i].plot(nn_preds[idx_set[i]], color='r', alpha=1, linewidth=1)
    axs[i].axhline(y=0.5, color='k', linestyle='--', linewidth=0.75, alpha=1)
    axs[i].axhline(y=0.3, color='green', linestyle='--', linewidth=0.75, alpha=1)
    axs[i].set_yticks([0, 0.5, 1])
    axs[i].set_xticks([])
    
fig.savefig('_'.join(str(n) for n in layers) + '.pdf')
plt.show()


def evaluate(y_test, nn_preds, threshold=0.5, topK=None):
    # given test set and predictions, compute and return micro-averaged P, R, and F1
    
    # test shapes of the two arrays
    assert y_test.shape == nn_preds.shape, "array shape mismatch."
    print("Computing performance measures ...")
    total_TPs = 0
    total_FPs = 0
    total_FNs = 0
    total_preds = 0
    total_trues = 0
    
    for idx in range(len(nn_preds)):
        if topK:
            # get the top K predictions based on the activation values of the output layer
            pred_labels = list(reversed(nn_preds[idx].ravel().argsort()[-topK:]))
        
        else:
            # get the predictions based on the given threshold
            pred_labels = np.where(nn_preds[idx] > threshold)[0].tolist()  
        
        true_labels = np.where(y_test[idx] == 1)[0].tolist()
        
        TPs = set(true_labels).intersection(pred_labels)
        FPs = set(pred_labels) - set(true_labels)
        FNs = set(true_labels) - TPs
        
        total_TPs += len(TPs)
        total_FPs += len(FPs)
        total_FNs += len(FNs)
     
    # micro-averaged statisitics
    avgP = total_TPs/(total_TPs + total_FPs)
    avgR = total_TPs/(total_TPs + total_FNs)
    
    # TODO: solve the division by zero exception
    try:
        F_score = (2*total_TPs)/(2*total_TPs + total_FPs + total_FNs)
    except:
        F_score = float('-inf')
    
    return (avgP, avgR, F_score)


# thresholding approach
P, R, F = evaluate(y_test, nn_preds, threshold=0.5)
print('Thresholded micro-averaged P, R, and F1')
print(P, R, F)

print('Top-K micro-averaged P, R, and F1')
P, R, F = evaluate(y_test, nn_preds, topK=3)
print(P, R, F)

