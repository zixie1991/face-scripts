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
    face_dir = args.face_dir
    faces = args.faces
    save_dir = args.save_dir
    num_map = int(args.num_map)

    map_reduce_split(faces, num_map)

    for i in range(num_map):
        os.system("""matlab -nodisplay -r "clear;face_dir='%s';faces='%s';save_dir='%s';face_5pt_align"& """ % (face_dir, map_name(faces, i), save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="batch process")
    parser.add_argument("face_dir", help="the face dir")
    parser.add_argument("faces", help="the faces")
    parser.add_argument("save_dir", help="the save_dir")
    parser.add_argument("--num_map", default=1, type=int, help="num map")
    args = parser.parse_args()

    main(args)
