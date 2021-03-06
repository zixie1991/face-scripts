#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score


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


def load_pairs(pairs_path):
    """load pairs.txt
    """
    print "Reading pairs.txt"
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)

    # assert(len(pairs) == 6000)
    return np.array(pairs)


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


def get_embeddings(embeddings, pair):
    if len(pair) == 3:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[0], pair[2].zfill(4))
        actual_same = True
    elif len(pair) == 4:
        name1 = "{}_{}".format(pair[0], pair[1].zfill(4))
        name2 = "{}_{}".format(pair[2], pair[3].zfill(4))
        actual_same = False
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


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def acc(predicts, thresholds=np.arange(-1.0, 1.0, 0.005)):
    print("...Computing accuracy.")
    folds = KFold(n=predicts.shape[0], n_folds=10, shuffle=False)
    accuracy = []
    thd = []
    for idx, (train, test) in enumerate(folds):
        print "processing fold {}...".format(idx)
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)

    return accuracy, thd
