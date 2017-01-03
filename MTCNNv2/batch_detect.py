#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

def map_name(filename, map_job):
    return filename + '-' + str(map_job)

def map_reduce_split(filename, num_map):
    with open(filename, 'r') as f:
        num_lines = len(f.readlines())

    avg_lines = (num_lines + num_map - 1) / num_map
    m = 0
    i = 0
    f = open(map_name(filename, m), 'w')
    with open(filename, 'r') as f2:
        for line in f2.readlines():
            f.write(line)
            i += 1
            if i == avg_lines:
                i = 0
                m += 1
                f.close()
                f = open(map_name(filename, m), 'w')


def main(args):
    root_folder = args.root_folder
    imglist = args.imglist
    labeled_faces = args.labeled_faces
    num_map = int(args.num_map)

    map_reduce_split(imglist, num_map)

    for i in range(num_map):
        os.system("""matlab -nodisplay -r "clear;root='%s';imglist='%s';labeled_faces='%s';face_detect_5pt"& """ % (root_folder, map_name(imglist, i), map_name(labeled_faces, i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="batch process")
    parser.add_argument("root_folder", help="the root folder")
    parser.add_argument("imglist", help="the imglist")
    parser.add_argument("labeled_faces", help="the labeled_faces")
    parser.add_argument("--num_map", default=1, type=int, help="num map")
    args = parser.parse_args()

    main(args)
