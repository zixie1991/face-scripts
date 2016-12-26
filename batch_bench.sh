#!/bin/bash

#snap_iters=(24000 26000 28000 30000 32000 34000 36000 38000 40000 42000)
snap_iters=(35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000)
#snap_iters=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 105000 110000)
for snap_iter in ${snap_iters[@]}; do
    python caffe/extract_feats.py ~/mydata/lfw-center-96/ ../face-recognition/examples/model_e7_6/deploy.prototxt ../face-recognition/examples/model_e7_6/snapshot/face_iter_${snap_iter}.caffemodel ./ --feat_layer fc6 >> test.log 2>&1 && python lfw/accuracy.py ./reps.csv ./labels.csv  --pairs ./lfw/pairs.txt --dist cosine >> test.log 2>&1 && python lfw/accuracy.py ./reps.csv ./labels.csv  --pairs ./lfw/pairs.txt --dist cosine --use_pca 128 >> test.log 2>&1
done
