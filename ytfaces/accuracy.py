#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from util import *


def main(args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("feat_file", help="the feat file")
    parser.add_argument("label_file", help="the label file")
    parser.add_argument("--splits_file", default="./ytfaces/splits.txt", help="splits.txt file path.")
    parser.add_argument("--dist", default="L2", help="dist function(L1, L2, cosine, Hamming).")
    parser.add_argument("--use_pca", default=0, type=int, help="use pca")
    args = parser.parse_args()

    main(args)
