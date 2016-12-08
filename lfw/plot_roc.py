#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import sys
import argparse

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from util import *


def write_ROC(embeddings, pairs, thresholds, roc_path, dist_func):
    fprs = []
    tprs = []
    with open(roc_path, 'w') as f:
        f.write('threshold,tp,tn,fp,fn,tpr,fpr\n')
        tp = tn = fp = fn = 0
        for threshold in thresholds:
            tp = tn = fp = fn = 0
            for pair in pairs:
                (x1, x2, actual_same) = get_embeddings(embeddings, pair)
                dist = dist_func(x1, x2)
                predict_same = dist < threshold

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


def eval_threshold_accuracy(embeddings, pairs, threshold, dist_func):
    y_true = []
    y_predict = []
    for pair in pairs:
        (x1, x2, actual_same) = get_embeddings(embeddings, pair)
        dist = dist_func(x1, x2)
        predict_same = dist < threshold
        y_predict.append(predict_same)
        y_true.append(actual_same)

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy


def find_best_threshold(embeddings, pairs, thresholds, dist_func):
    best_thresh = best_thresh_acc = 0
    for threshold in thresholds:
        accuracy = eval_threshold_accuracy(embeddings, pairs, threshold, dist_func)
        if accuracy >= best_thresh_acc:
            best_thresh_acc = accuracy
            best_thresh = threshold
        else:
            # No further improvements.
            return best_thresh

    return best_thresh


def compute_accuracy(embeddings, pairs, output_dir, dist_func=L2_dist):
    print 'Compute accuracy'
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(0, 10, 0.1)

    accuracies = []
    acc_path = os.path.join(output_dir, 'accuracies.txt')
    fs = []
    with open(acc_path, 'w') as f:
        f.write('fold, threshold, accuracy\n')
        for idx, (train, test) in enumerate(folds):
            roc_path = os.path.join(output_dir, "roc.fold-{}.csv".format(idx))
            fprs, tprs = write_ROC(embeddings, pairs[test], thresholds, roc_path, dist_func)
            fs.append(interp1d(fprs, tprs))

            best_thresh = find_best_threshold(embeddings, pairs[train], thresholds, dist_func)
            accuracy = eval_threshold_accuracy(embeddings, pairs[test], best_thresh, dist_func)
            accuracies.append(accuracy)
            f.write('{}, {:0.2f}, {:0.2f}\n'.format(idx, best_thresh, accuracy))

        avg = np.mean(accuracies)
        std = np.std(accuracies)
        f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
        print('    + {:0.4f}'.format(avg))

    return fs


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


class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg


def usage():
    print "Usage: %s --feats=<feats> --labels=<labels> --pairs=<pairs.txt> --output_dir=<output_dir>" % (sys.argv[0])


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
        "--output_dir",
        default="./",
        help="output dirname."
    )
    parser.add_argument(
        "--dist",
        default="L2",
        help="dist function(L1, L2, cosine, Hamming)."
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
    pairs = load_pairs(args.pairs_file)
    fs = compute_accuracy(embeddings, pairs, args.output_dir, dist)
    plot_verify(fs, 'self model', args.output_dir)


if __name__ == '__main__':
    main()
