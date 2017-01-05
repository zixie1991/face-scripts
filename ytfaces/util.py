#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA


def normalize(x):
    '''向量归一化
    '''
    return x / np.linalg.norm(x)


def int_to_8bit_list(x):
    """init to list, element is in [0, 255]
    """
    for i in range(len(x)):
        x[i] = int(x[i])

    y = [0 for i in range(len(x) * 8)]
    i = 0
    for v in x:
        for j in range(8):
            if v > 0:
                y[i * 8 + j] = v % 2
                v /= 2
        i += 1

    return y


def Hamming_dist(x, y):
    # print len(x), x
    x = int_to_8bit_list(x)
    # print len(x), x
    y = int_to_8bit_list(y)
    distance=0
    for i in range(len(x)):
        if x[i]!=y[i]:
            distance+=1

    return distance


def L2_dist(x, y):
    '''欧氏距离
    '''
    return np.linalg.norm(np.array(normalize(x)) - np.array(normalize(y)))
    # return np.linalg.norm(np.array(x) - np.array(y))
    # return np.array(x).dot(np.array(y))


def L1_dist(x, y):
    '''城市区间距离
    '''
    return np.linalg.norm(np.array(x) - np.array(y), 1)


def cosine_dist(x, y):
    # dist = np.linalg.norm(np.array(x) - np.array(y))
    # sim = 1.0 / (1.0 + dist)  # 归一化

    sim = np.dot(np.array(x), np.array(y)) / (np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(y)))

    return sim


def load_splits(splits_path):
    """load splits.txt
    """
    print "Reading splits.txt"
    splits = []
    with open(splits_path, 'r') as f:
        for line in f.readlines()[1:]:
            split = line.strip().split(',')[2:]
            split = [x.strip() for x in split]
            if len(split) == 4:
                del split[-2]
            splits.append(split)

    return splits


def load_embeddings(feats_path, labels_path):
    embeddings = {}
    with open(feats_path, 'r') as f:
        feats = []
        for line in f.readlines():
            feats.append([float(x) for x in line.strip().split()])

    with open(labels_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip().split('/')
            line = [x.strip() for x in line]
            path = '/'.join(line[-3:-1])
            if path in embeddings:
                if len(embeddings[path]) >= 100:
                    continue
                embeddings[path].append(feats[idx])
            else:
                embeddings[path] = [feats[idx]]

    for path in embeddings:
        embeddings[path] = np.array(embeddings[path])
        embeddings[path] = embeddings[path].mean(axis=0)

    return embeddings


def get_embeddings(embeddings, pair):
    if len(pair) == 3:
        name1 = pair[0]
        name2 = pair[1]
        actual_same = int(pair[2])
    else:
        raise Exception(
            "Unexpected pair length: {}".format(len(pair)))

    (x1, x2) = (embeddings[name1], embeddings[name2])
    return (x1, x2, actual_same)


def pca(data, n_components=128):
    pca = PCA(n_components=n_components)
    pca.fit(data)

    result = pca.transform(data)

    return result
