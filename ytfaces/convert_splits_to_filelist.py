#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse


def pairs_to_files(path):
    file_idx = {}
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(',')
            if len(line) == 3:
                yield '{}/{}_{}.jpg'.format(line[0], line[0], line[1].zfill(4))
                yield '{}/{}_{}.jpg'.format(line[0], line[0], line[2].zfill(4))
            elif len(line) == 4:
                yield '{}/{}_{}.jpg'.format(line[0], line[0], line[1].zfill(4))
                yield '{}/{}_{}.jpg'.format(line[2], line[2], line[3].zfill(4))


def main(args):
    files = set()
    with open(args.dest, 'w') as f:
        for line in pairs_to_files(args.pairs):
            if line in files:
                continue
            files.add(line)
            f.write(line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract feats")
    parser.add_argument("pairs", help="the pairs file")
    parser.add_argument("dest", help="the dest file")
    args = parser.parse_args()
    main(args)
