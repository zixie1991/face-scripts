#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import numpy as np

from utils import allowed_file


brew_func_map = {}


def face_file(filename):
    return True

brew_func_map['face'] = face_file


def brow_file(filename):
    if filename.startswith('left_brow') or filename.startswith('right_brow'):
        return True

    return False

brew_func_map['brow'] = brow_file


def eye_file(filename):
    if filename.startswith('left_eye') or filename.startswith('right_eye'):
        return True

    return False

brew_func_map['eye'] = eye_file


def nose_tip_file(filename):
    if filename.startswith('nose_tip'):
        return True

    return False

brew_func_map['nose_tip'] = nose_tip_file


def mouth_file(filename):
    if filename.startswith('left_mouth') or filename.startswith('right_mouth'):
        return True

    return False

brew_func_map['mouth'] = mouth_file


def create_train_val(input_dir, output_dir, val_ratio=0.05, face_type='face', min_faces=1, num_persons=0, sort=False):
    dir_idx = 0
    train_f = open(os.path.join(output_dir, 'train.txt'), 'w')
    val_f = open(os.path.join(output_dir, 'val.txt'), 'w')
    while input_dir[-1] == '/':
        input_dir = input_dir[:-1]

    dir_walk = {}
    for root, dirs, files in os.walk(input_dir):
        if root == './':
            continue
        if dirs:
            continue

        num_files = len(files)
        if num_files < min_faces:
            continue

        dir_walk[root] = (num_files, files)

    if sort:
        dir_walk = sorted(dir_walk.iteritems(), key=lambda v: v[1][0], reverse=True)
    else:
        dir_walk = dir_walk.items()

    if num_persons == 0:
        num_persons = len(dir_walk)

    if sort:
        data_idx = np.arange(len(dir_walk))
    else:
        data_idx = np.random.choice(range(len(dir_walk)), num_persons, replace=False)

    dir_idx = -1
    person_idx = 0
    for root, (num_files, files) in dir_walk:
        dir_idx += 1

        if person_idx >= num_persons:
            break

        if dir_idx not in data_idx:
            continue

        dir_name = root[len(input_dir):]
        while dir_name[0] == '/':
            dir_name = dir_name[1:]

        num_vals = int(num_files * val_ratio) if num_files * val_ratio > 1 else 1
        val_idx = np.random.choice(np.arange(len(files)), num_vals, replace=False)

        for idx, filename in enumerate(files):
            if not allowed_file(filename):
                continue
            if not brew_func_map[face_type](filename):
                continue

            file_path = os.path.join(dir_name, filename)
            if idx in val_idx:
                val_f.write(file_path + ' ' + str(person_idx) + '\n')
                # val_f.write(file_path + '\t' + str(person_idx) + '\n')
            else:
                train_f.write(file_path + ' ' + str(person_idx) + '\n')
                # train_f.write(file_path + '\t' + str(person_idx) + '\n')

        person_idx += 1


def main(argv=None):
    """
    整个数据集做为人脸集
    python create_train_val.py input_dir output_dir --min_faces 10
    生成不同规模的人脸集
    python create_train_val.py input_dir output_dir --num_person 1000 --min_faces 10
    按人脸数排序的人脸集(逆序)
    python create_train_val.py input_dir output_dir --num_person 1000 --sort yes
    """
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description="Face align benchmark")
    # Required arguments: input and output files.
    parser.add_argument(
        "input_dir",
        help="input dir."
    )
    parser.add_argument(
        "output_dir",
        default="./",
        help="output dir(default: ./)."
    )
    parser.add_argument(
        "--val_ratio",
        default=0.00001,
        type=float,
        help="val ratio in every dir(type: int, default: 0.00001)"
    )
    parser.add_argument(
        "--face_type",
        default="face",
        help="face type(include: face, brow, eye, nose_tip, mouth, default: face)"
    )
    parser.add_argument(
        "--min_faces",
        type=int,
        default=1,
        help="min num of faces(type: int, default: 1)"
    )
    parser.add_argument(
        "--num_persons",
        type=int,
        default=0,
        help="num of persons(type: int, default: 0)"
    )
    parser.add_argument(
        "--sort",
        default="no",
        help="select persons by sorted faces of every person(include: yes, no, default: no)"
    )

    args = parser.parse_args()

    sort = False
    if args.sort == 'yes':
        sort = True

    create_train_val(args.input_dir, args.output_dir, val_ratio=args.val_ratio, face_type=args.face_type, min_faces=args.min_faces, num_persons=args.num_persons, sort=sort)

if __name__ == '__main__':
    main()
