#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from util import *


def get_predicts(embeddings, pairs, dist_func=L2_dist):
    '''读取数据集
    '''
    f = open('predicts.txt', 'w')
    predicts = []
    for pair in pairs:
        try:
            (x1, x2, actual_same) = get_embeddings(embeddings, pair)
        except:
            continue
        dist = dist_func(x1, x2)
        predicts.append([dist, 1 if actual_same else 0])
        # print pair, dist
        # f.write(' '.join(pair) + ' ' + dist + ' ' + actual_same + '\n')
        # print pair, dist
    return predicts

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

    predicts2 = []
    for i in idx:
        predicts2.append(predicts[idx[i]])

    return predicts2


def main(args):
    if args.dist == 'Hamming':
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

    pairs = load_pairs(args.pairs_file)
    predicts = get_predicts(embeddings, pairs, dist_func=dist)
    predicts = np.array(predicts)

    accuracy, threshold = acc(predicts)
    print "10-fold accuracy is:\n{}\n".format(accuracy)
    print "10-fold threshold is:\n{}\n".format(threshold)
    print "mean threshold is:%.4f\n", np.mean(threshold)
    print "mean is:%.4f, var is:%.4f", np.mean(accuracy), np.std(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("feat_file", help="the feat file")
    parser.add_argument("label_file", help="the label file")
    parser.add_argument("--pairs_file", default="./lfw/pairs.txt", help="pairs.txt file path.")
    parser.add_argument("--dist", default="L2", help="dist function(cosine, Hamming).")
    parser.add_argument("--use_pca", default=0, type=int, help="use pca")
    args = parser.parse_args()

    main(args)
