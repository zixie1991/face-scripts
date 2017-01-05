#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os


def splits_to_files(root, path):
    files = set()
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')

            for parent_path in line[-3:-1]:
                parent_path = parent_path.strip()
                for filename in os.listdir(os.path.join(root, parent_path))[:100]:

                    filename = os.path.join(parent_path, filename)
                    if filename not in files:
                        files.add(filename)
                        yield filename


def main(args):
    files = set()
    with open(args.dest, 'w') as f:
        for line in splits_to_files(args.root, args.splits):
            if line in files:
                continue
            files.add(line)
            f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert splits to file list")
    parser.add_argument("root", help="the root folder")
    parser.add_argument("splits", help="the splits file")
    parser.add_argument("dest", help="the dest file")
    args = parser.parse_args()
    main(args)
