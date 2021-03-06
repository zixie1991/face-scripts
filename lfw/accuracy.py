#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import random

from sklearn import svm
from sklearn import cross_validation

from util import *


def get_dataset(embeddings, pairs, dist_func=L2_dist):
    '''读取数据集
    '''
    X = []
    y = []
    for pair in pairs:
        try:
            (x1, x2, actual_same) = get_embeddings(embeddings, pair)
        except:
            continue
        dist = dist_func(x1, x2)
        X.append([dist])
        y.append(1 if actual_same else 0)
        # print pair, dist
    return X, y

    # shuffle data
    t1 = range(len(X) / 2)
    random.shuffle(t1)
    t2 = range(len(X) / 2, len(X))
    random.shuffle(t2)
    i = 0
    j = 0
    idx = []
    while i < len(t1) or j < len(t2):
        if i < len(t1):
            idx.append(t1[i])
            i += 1

        if j < len(t2):
            idx.append(t2[j])
            j += 1

    X2 = []
    y2 = []
    for i in idx:
        X2.append(X[idx[i]])
        y2.append(y[idx[i]])

    return X2, y2


def verification_cross_validation(samples, labels, n_iter=10, test_size=0.1):
    '''交叉验证
    默认为 10-fold cross validation(10折交叉验证)
    '''
    classifier = svm.SVC()
    # classifier = svm.LinearSVC()
    num_sample = len(labels)

    cv = cross_validation.KFold(num_sample, n_folds=10, shuffle=False)
    # cv = cross_validation.ShuffleSplit(num_sample, n_iter=n_iter, test_size=test_size, random_state=0)

    scores = cross_validation.cross_val_score(classifier, samples, labels, cv=cv)
    print 'Cross validation scores', scores

    print 'Cross validation score %f +/- %f' % (np.mean(np.array(scores)), np.std(np.array(scores)))


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description="LFW face verification")
    # Required arguments: input and output files.
    parser.add_argument(
        "feats_file",
        help="feats file path."
    )
    parser.add_argument(
        "labels_file",
        help="labels file path."
    )
    # Optional arguments.
    parser.add_argument(
        "--pairs_file",
        default="./pairs.txt",
        help="pairs.txt file path."
    )
    parser.add_argument(
        "--dist",
        default="L2",
        help="dist function(L1, L2, cosine, Hamming)."
    )
    parser.add_argument(
        "--use_pca",
        default=0,
        type=int,
        help="use pca"
    )

    args = parser.parse_args()
    if args.dist == 'L2':
        dist = L2_dist
    elif args.dist == 'L1':
        dist = L1_dist
    elif args.dist == 'Hamming':
        dist = Hamming_dist
    elif args.dist == "cosine":
        dist = cosine_dist

    embeddings = load_embeddings(args.feats_file, args.labels_file)
    if args.use_pca:
        data = []
        keys = sorted(embeddings.keys())
        for k in keys:
            data.append(embeddings[k])

        print len(data[0])
        data = pca(data, n_components=args.use_pca)
        print len(data[0])
        for i, k in enumerate(keys):
            embeddings[k] = data[i]

    pairs = load_pairs(args.pairs_file)
    X, y = get_dataset(embeddings, pairs, dist_func=dist)
    verification_cross_validation(X, y)


if __name__ == '__main__':
    main()
