#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import os.path
import caffe
import argparse
import cv2
import sys

sys.path.append('/home/zixie1991/mygit/caffe/python')


class Predictor(object):

    def __init__(self, param, weights, mode=-1):
        self._net = caffe.Net(param, weights, caffe.TEST)
        if mode < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(mode)

    def forward(self, img, output_layer_name):
        img = self.transform(img)
        data = np.array([img], dtype=np.float32)
        self._net.set_input_arrays(data, np.array([0], dtype=np.float32))
        out = self._net.forward(blobs=[output_layer_name])

        # out = net.forward(data=img[np.newaxis, ...], blobs=[layer])
        return out[output_layer_name]

    def transform(self, img):
        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        img = img.astype(np.float32)
        # img -= mean_value
        # img *= scale
        img -= 127.5
        img /= 128.0

        img = img.transpose(2, 0, 1)
        return img

def gen_image_path(root):
    for folder in os.listdir(root):
        folder_abs = os.path.join(root, folder)
        if not os.path.isdir(folder_abs):
            continue
        for img in os.listdir(folder_abs):
            abs_path = os.path.join(root, folder, img)
            rel_path = os.path.join(folder, img)
            yield rel_path, abs_path


def write_to_file(ffeats, flabels, feats, labels):
    flabels.write("%s\n" % labels)
    f = [str(i) for i in feats]
    ffeats.write("%s\n" % ' '.join(f))


def main(args):
    root = args.root
    model = args.model
    proto = args.proto
    dest = args.dest

    feats_fname = os.path.join(dest, "reps.csv")
    label_fname = os.path.join(dest, "labels.csv")

    if not os.path.isdir(dest):
        os.mkdir(dest)
    ffeats = open(feats_fname, "w")
    flabels = open(label_fname, "w")

    predictor = Predictor(proto, model, args.mode)

    labels = []
    print("begin")
    for rel_path, abs_path in gen_image_path(root):
        # img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(abs_path)
        labels = rel_path
        feats = predictor.forward(img, args.feat_layer)
        bc_feat = []
        for i in range(feats.shape[1]):
            bc_feat.append(1 if feats[0][i] > 0.5 else 0)
        write_to_file(ffeats, flabels, bc_feat, labels)
    ffeats.close()
    flabels.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract feats")
    parser.add_argument("root", help="the image root dir")
    parser.add_argument("proto", help="the proto file")
    parser.add_argument("model", help="the model file")
    parser.add_argument("dest", help="the dest folder")
    parser.add_argument("--feat_layer", "-f", default="fc5", help="feat layer")
    parser.add_argument("--mode", "-m", default=0, type=int, help="mode")
    args = parser.parse_args()
    main(args)
