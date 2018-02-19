# coding: utf-8

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# read options from user input, and generate dataset.
DISCR = 'Analyse and produce stats from word clusters.'
parser = argparse.ArgumentParser(description=DISCR)
parser.add_argument('-input', type=str, help='Path to input file.', required=True)
parser.add_argument('-output', type=str, help='Path to output files.', required=True)


#parser.add_argument('-word_output', type=str, 
#	help='Path to words pers cluser file name.', required=True)
#parser.add_argument('-oov_output', type=str, 
#	help='Path to OOVs pers cluser file name.', required=True)

args = parser.parse_args()

# read from file 
# '/home/babdulla/DNN/wiki-processed/wiki-clusters.1k'

print('Reading word clusters from disk ...')
with open(args.input) as txtFile:
    lines = txtFile.readlines()


cluster2words = defaultdict(list)


for line in lines:
    word, cluster = line.split()
    cluster2words[int(cluster)].append(word)


# writing docs to disk 
print('Writing processed words per cluster to disk ...')
for c in cluster2words:
    with open(args.output + 'words_per_cluster', 'a+', encoding="utf8") as oFile:
        oFile.write(' '.join(cluster2words[c]) + '\n')

# analyse length distribution
len_dist = {}

for (c, w) in cluster2words.items():
    len_dist[c] = len(w)

# obtian some stats 
n_words = np.array(list(len_dist.values()))

print("Mean count:       ", n_words.mean())
print("Standard deviation:", n_words.std())
print("Minimum count:    ", n_words.min())
print("Maximum count:    ", n_words.max())
print("25th percentile:   ", np.percentile(n_words, 25))
print("Median:            ", np.median(n_words))
print("75th percentile:   ", np.percentile(n_words, 75))

plt.bar(len_dist.keys(), sorted(len_dist.values(), reverse=True))
plt.title('Distribution of cluster sizes')
plt.xlabel('Cluster ID')
plt.ylabel('number of words')
#plt.ylim(0, 2500)
plt.show()


# analyse OOVs
OOV_clusters = defaultdict(list)

for c in cluster2words:
    OOV_clusters[c] = [w for w in cluster2words[c] if w.find('OOV') != -1]
    # print(cluster2words[c][:20])

# writing docs to disk 
print('Writing processed OOvs per cluster to disk ...')
for c in cluster2words:
    with open(args.output + 'OOVs_per_cluster', 'a+', encoding="utf8") as oFile:
        oFile.write(' '.join(OOV_clusters[c]) + '\n')

OOV_dist = {}

for (c, oov) in OOV_clusters.items():
    OOV_dist[c] = len(oov)

n_OOVs = np.array(list(OOV_dist.values()))

print("Mean count:       ", n_OOVs.mean())
print("Standard deviation:", n_OOVs.std())
print("Minimum count:    ", n_OOVs.min())
print("Maximum count:    ", n_OOVs.max())
print("25th percentile:   ", np.percentile(n_OOVs, 25))
print("Median:            ", np.median(n_OOVs))
print("75th percentile:   ", np.percentile(n_OOVs, 75))

