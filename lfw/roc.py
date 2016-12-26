#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import math
import sys

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from util import *


def get_predicts(embeddings, pairs, dist_func=L2_dist):
    '''读取数据集
    '''
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


def write_ROC(predicts, thresholds, roc_path):
    fprs = []
    tprs = []
    with open(roc_path, 'w') as f:
        f.write('threshold,tp,tn,fp,fn,tpr,fpr\n')
        tp = tn = fp = fn = 0
        for threshold in thresholds:
            tp = tn = fp = fn = 0
            for d in predicts:
                predict_same = d[0] > threshold
                actual_same = d[1]

                if predict_same and actual_same:
                    tp += 1
                elif predict_same and not actual_same:
                    fp += 1
                elif not predict_same and not actual_same:
                    tn += 1
                elif not predict_same and actual_same:
                    fn += 1

            if tp + fn == 0:
                tpr = 0
            else:
                tpr = float(tp) / float(tp + fn)
            if fp + tn == 0:
                fpr = 0
            else:
                fpr = float(fp) / float(fp + tn)

            f.write(','.join([str(x) for x in [threshold, tp, tn, fp, fn, tpr, fpr]]))
            f.write('\n')
            if tpr == 1.0 and fpr == 1.0:
                # No further improvements.
                f.write(','.join([str(x) for x in [4.0, tp, tn, fp, fn, tpr, fpr]]))

            fprs.append(fpr)
            tprs.append(tpr)

    return fprs, tprs


def get_AUC(fprs, tprs):
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def plot_ROC(fs, color=None):
    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)

    if color:
        mean_plot, = plt.plot(fprs, tprs, color=color)
    else:
        mean_plot, = plt.plot(fprs, tprs)

    AUC = get_AUC(fprs, tprs)

    return mean_plot, AUC


def plot_verify(fs, tag, output_dir):
    print 'Plotting.'

    fig, ax = plt.subplots(1, 1)

    mean_plot, AUC = plot_ROC(fs)

    ax.legend([mean_plot], ['{}({})'.format(tag, AUC)], loc='lower right')
    plt.plot([0, 1], color='k', linestyle=':')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(output_dir, "roc.pdf"))
    fig.savefig(os.path.join(output_dir, "roc.png"))


def compute_accuracy(predicts, output_dir):
    print 'Compute accuracy'
    folds = KFold(n=predicts.shape[0], n_folds=10, shuffle=False)
    thresholds = np.arange(1.0, -1.0, -0.005)

    accuracies = []
    acc_path = os.path.join(output_dir, 'accuracies.txt')
    fs = []
    with open(acc_path, 'w') as f:
        f.write('fold, threshold, accuracy\n')
        for idx, (train, test) in enumerate(folds):
            roc_path = os.path.join(output_dir, "roc.fold-{}.csv".format(idx))
            fprs, tprs = write_ROC(predicts[test], thresholds, roc_path)
            fs.append(interp1d(fprs, tprs))
            # fs.append(interp1d(fprs, tprs, bounds_error=False))

            best_thresh = find_best_threshold(thresholds, predicts[train])
            accuracy = eval_acc(best_thresh, predicts[test])
            accuracies.append(accuracy)
            f.write('{}, {:0.2f}, {:0.2f}\n'.format(idx, best_thresh, accuracy))

        avg = np.mean(accuracies)
        std = np.std(accuracies)
        f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
        print('    + {:0.4f}'.format(avg))

    return fs


def main(args):
    if args.dist == 'Hamming':
        dist = Hamming_dist
    elif args.dist == "cosine":
        dist = cosine_dist

    embeddings = load_embeddings(args.feat_file, args.label_file)
    pairs = load_pairs(args.pairs_file)
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

    predicts = get_predicts(embeddings, pairs, dist_func=dist)
    predicts = np.array(predicts)
    fs = compute_accuracy(predicts, args.output_dir)
    plot_verify(fs, 'self model', args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("feat_file", help="the feat file")
    parser.add_argument("label_file", help="the label file")
    parser.add_argument("--output_dir", default="./", help="output dirname.")
    parser.add_argument("--pairs_file", default="./lfw/pairs.txt", help="pairs.txt file path.")
    parser.add_argument("--dist", default="L2", help="dist function(cosine, Hamming).")
    parser.add_argument("--use_pca", default=0, type=int, help="use pca")
    args = parser.parse_args()

    main(args)
