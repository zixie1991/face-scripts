#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse


def main(args):
    rectlist = {}
    with open(args.rectlist, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            rectlist[line[0]] = [int(line[2]), int(line[3]), int(line[4]), int(line[5])]

    f2 = open(args.dest, 'w')
    with open(args.f5ptlist, 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break

            path, numbox = line.strip().split(' ')
            numbox = int(numbox)
            line = f.readline()
            line = line.strip().split(' ')
            f5pts = []
            for i in range(numbox):
                f5pts.append(line[10 * i: 10 * (i + 1)])

            line = f.readline()
            line = line.strip().split(' ')
            dist = 1e8
            j = -1
            rect = rectlist[path]
            for i in range(numbox):
                d = (rect[0] - (float(line[i * 4 + 0]) + float(line[i * 4 + 2]))/2.) ** 2 + (rect[1] - (float(line[i * 4 + 1]) + float(line[i * 4 + 3]))/2.) ** 2
                if d < dist:
                    dist = d
                    j = i

            f2.write(path + ' 1 ' + ' '.join(f5pts[j]) + '\n')

        f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rebuild labeled faces")
    parser.add_argument("rectlist", help="the face rect file list")
    parser.add_argument("f5ptlist", help="the face 5pt file list")
    parser.add_argument("dest", help="the dest folder")
    args = parser.parse_args()

    main(args)
