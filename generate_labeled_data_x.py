# coding: utf-8

import matplotlib.pyplot as plt
from collections import defaultdict, Counter 
import numpy as np

print('Reading vocab from disk ...')
with open('/home/babdulla/DNN/wiki-processed/vocab.96k.amelie') as txtFile:
    vocab = [w.strip() for w in txtFile.readlines()]

print('Reading wikipedia docs from disk ...')
with open('/home/babdulla/DNN/wiki-processed/wikipedia.cleaned.nt') as txtFile:
    documents = txtFile.readlines()

vocab = set(vocab)

# read clusters
print('Reading word clusters from disk ...')
with open('/home/babdulla/DNN/wiki-processed/wiki-clusters.1k') as txtFile:
    lines = txtFile.readlines()
        

word2cluster = defaultdict(lambda: -1)

for line in lines:
    word, cluster = line.split()
    word2cluster[word] = cluster

docs_labeled = list()

for i in range(len(documents)): # [:50000]
    doc_IVs = list(); OOV_PNs = list(); clusters = list()
    
    if (i+1)%10000 == 0: print(i+1, ' docs processed so far.')
    
    for w in documents[i].split():
        # if IV then append to the doc IV list 
        if w in vocab:
            doc_IVs.append(w)
           
        elif w.find('OOV') != -1:
            clusters.append(int(word2cluster[w]))
            OOV_PNs.append(w)

    # add only if there are OOV_PNs in the doc    
    if OOV_PNs:
        docs_labeled.append((doc_IVs, OOV_PNs, clusters))


doc_words    = [doc[0] for doc in docs_labeled]
doc_OOV_PNs  = [doc[1] for doc in docs_labeled]
doc_clusters = [set(doc[2]) for doc in docs_labeled]


print('Writing cleaned data to disk ...')
with open('/home/babdulla/DNN/wiki-data/wikidocs.csv', mode="w") as oFile:
    oFile.write('words,clusters\n')
    for (W, C) in zip(doc_words, doc_clusters):
        oFile.write(' '.join(W) + ',' + ' '.join(str(c) for c in C) + '\n')

print('Computing some stats ...')
n_labels = [len(set(l_list)) for l_list in doc_clusters]
n_labels = np.array(n_labels)

print("Mean count:       ", n_labels.mean())
print("Standard deviation:", n_labels.std())
print("Minimum count:    ", n_labels.min())
print("Maximum count:    ", n_labels.max())
print("25th percentile:   ", np.percentile(n_labels, 25))
print("Median:            ", np.median(n_labels))
print("75th percentile:   ", np.percentile(n_labels, 75))


plt.hist(n_labels, bins=100)
plt.title('Distribution of number of labels')
plt.xlabel('number of classes')
plt.ylabel('number of documents')
plt.show()

