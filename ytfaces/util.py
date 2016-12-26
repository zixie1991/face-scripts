#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA


def load_splits(splits_path):
    """load splits.txt
    """
    print "Reading splits.txt"
    splits = []
    with open(splits_path, 'r') as f:
        for line in f.readlines()[1:]:
            split = line.strip().split()[2:]
            splits.append(split)


def load_embeddings(feats_path, labels_path):
    embeddings = {}
    with open(feats_path, 'r') as f:
        feats = []
        for line in f.readlines():
            feats.append([float(x) for x in line.strip().split()])

    with open(labels_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            embeddings[line.strip().split('/')[-1].split('.')[-2]] = feats[idx]

    return embeddings


def pca(data, n_components=128):
    pca = PCA(n_components=n_components)
    pca.fit(data)

    result = pca.transform(data)

    return result
