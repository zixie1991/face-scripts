#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

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
        print pair, dist
    return X, y


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


def main(args):
    args = parser.parse_args()
    if args.dist == 'L2':
        dist = L2_dist
    elif args.dist == 'L1':
        dist = L1_dist
    elif args.dist == 'Hamming':
        dist = Hamming_dist
    elif args.dist == "cosine":
        dist = cosine_dist

    embeddings = load_embeddings(args.feat_file, args.label_file)
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

    splits = load_splits(args.splits_file)
    X, y = get_dataset(embeddings, splits, dist_func=dist)

    verification_cross_validation(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("feat_file", help="the feat file")
    parser.add_argument("label_file", help="the label file")
    parser.add_argument("--splits_file", default="./ytfaces/splits.txt", help="splits.txt file path.")
    parser.add_argument("--dist", default="L2", help="dist function(L1, L2, cosine, Hamming).")
    parser.add_argument("--use_pca", default=0, type=int, help="use pca")
    args = parser.parse_args()

    main(args)
