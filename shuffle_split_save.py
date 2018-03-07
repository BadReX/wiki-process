# coding: utf-8

# basic imports
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import math 
import sys
import os
import time

# other imports
from sklearn.utils import shuffle


print('Reading data from disk ...')
wikidocs = pd.read_csv('/home/babdulla/DNN/wiki-data/wikidocs.csv', sep=',')
print('Dimensions of data frame (r, c):', wikidocs.shape)

wikidocs_shuffled = shuffle(wikidocs)

assert wikidocs.shape == wikidocs_shuffled.shape, "Data shape mismatch."


# save shuffled dataframe to disk
wikidocs_shuffled.to_csv('/home/babdulla/DNN/wiki-data/wikidocs.shuffled.csv', 
                         encoding='utf-8', 
                         sep=',', 
                        columns=['words', 'clusters'],
                        index=False)

# make train, dev, and test splits
print('Making train, dev, and test splits ...')  
DEV_SPLIT  = 0.1
TEST_SPLIT = 0.1

ndev_samples  = int(DEV_SPLIT * wikidocs_shuffled.shape[0])
ntest_samples = int(TEST_SPLIT * wikidocs_shuffled.shape[0])


dev_idx = -ndev_samples - ntest_samples

train_split = wikidocs_shuffled[:dev_idx]

dev_split = wikidocs_shuffled[dev_idx:-ndev_samples]

test_split = wikidocs_shuffled[-ndev_samples:]


print('Dimensions of the splits', train_split.shape, dev_split.shape, test_split.shape)

# save split dataframes to disk 
train_split.to_csv('/home/babdulla/DNN/wiki-data/wikidocs.train.csv', 
					 encoding='utf-8', 
					 sep=',', 
					 columns=['words', 'clusters'],
					 index=False)

dev_split.to_csv('/home/babdulla/DNN/wiki-data/wikidocs.dev.csv', 
					encoding='utf-8', 
					sep=',', 
					columns=['words', 'clusters'],
					index=False)

test_split.to_csv('/home/babdulla/DNN/wiki-data/wikidocs.test.csv', 
					encoding='utf-8', 
					sep=',', 
					columns=['words', 'clusters'],
					index=False)

