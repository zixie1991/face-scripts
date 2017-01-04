#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil


def main(args):
    with open(args.imglist, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            src_path = os.path.join(args.root, line)
            dst_path = os.path.join(args.dest, line)
            if not os.path.exists(os.path.split(dst_path)[0]):
                os.makedirs(os.path.split(dst_path)[0])
            shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract feats")
    parser.add_argument("root", help="the image root dir")
    parser.add_argument("imglist", help="the image list file")
    parser.add_argument("dest", help="the dest folder")
    args = parser.parse_args()
    main(args)
