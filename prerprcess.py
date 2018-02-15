# coding: utf-8

from collections import Counter 
from scipy import stats, integrate
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

with open('/home/babdulla/DNN/wikiall/wikipedia.semi-cleaned.nt') as txtFile:
    wiki_text = txtFile.readlines()

# segment by sentence 
# this data structure is as list of tuples (doc_str, doc_len)
# [(doc_1, len(doc_1)), (doc_2, len(doc_2), ..., (doc_N, len(doc_N)]

docs = list()

for doc in wiki_text:    
    # ignore the first title sentence
    sentences = doc.split('||')[1:]
    
    # obtain length for each doc by number of words
    doc_length = sum([len(s.split()) for s in sentences if len(s.split()) > 0])
    
    # if doc is too short, do not bother
    if doc_length < 10:
        print(doc)
        continue 
        
    docs.append((' '.join(sentences).strip(), doc_length))

# make sure everything is right
for doc in docs:
    assert doc[1] == len(doc[0].split()), 'Length mismatch'

def make_segments(max_len):
    """
    A function that takes a list of docs and returns lists of segments 
    less than a given threshold (measure by the number of words)
    """
    segments = list()
    buffer = ''
    
    for d, d_len in docs:   
        
        if d_len < max_len:
            segments.append(d)
            
        else:
            # if the length of the doc is larger than 500, make many segments
            N = math.floor(d_len/max_len + 1)  # number of segments 
            s_len = d_len/N # length of each segment by number of words
            
            # tokenize each doc
            d = d.split()
            
            for i in range(N):
                str_idx = int(i*s_len)
                end_idx = int((i+1)*s_len)
                
                # de-tokenize then append
                segments.append(' '.join(d[str_idx:end_idx]))
                
    return segments              


segments = make_segments(499)


lengths = list()

"""
for s in segments:
    l = len(s.split())
    
    if l < 40:
        print('-->', l, s)



for d, l in docs:

    if l < 10:
        print('-->', l, d)


"""

data = [len(s.split()) for s in segments]
lengths = np.array(data)

print("Mean lengths:       ", lengths.mean())
print("Standard deviation:", lengths.std())
print("Minimum lengths:    ", lengths.min())
print("Maximum lengths:    ", lengths.max())
print("25th percentile:   ", np.percentile(lengths, 25))
print("Median:            ", np.median(lengths))
print("75th percentile:   ", np.percentile(lengths, 75))

plt.hist(lengths, bins=500)
plt.title('Length Distribution of Segments')
plt.xlabel('length (words)')
plt.ylabel('number of segments')
plt.show()

