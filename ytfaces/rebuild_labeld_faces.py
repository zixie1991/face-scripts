#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os


def rebuild_face(line):
    line = line.strip()  # 去除两边空白
    path, _, x, y, w, h, _, _ = line.split(',')

    path = '/'.join(path.split('\\'))
    # l = x
    # t = y
    # r = str(int(x) + int(w))
    # b = str(int(y) + int(h))

    # return path + ' 1 ' + l + ' ' + t + ' ' + r + ' ' + b
    return path + ' 1 ' + x + ' ' + y + ' ' + w + ' ' + h


def rebuild_labeled_faces(filepath):
    with open(filepath, 'r') as f:
        for line in f.readlines():
            yield rebuild_face(line)


def main(args):
    save_file = open(args.dest, 'w')
    for name in os.listdir(args.root):
        if not name.endswith('.labeled_faces.txt'):
            continue

        if os.path.isfile(os.path.join(args.root, name)):
            for face in rebuild_labeled_faces(os.path.join(args.root, name)):
                save_file.write(face + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("root", help="the image root dir")
    parser.add_argument("dest", help="the dest folder")
    args = parser.parse_args()

    main(args)
