#!/bin/bash

wget https://www.dropbox.com/s/c3h9e0o6ajydbao/clothing-small.zip

unzip -uq 'clothing-small.zip' -d '.'
rm clothing-small.zip
mv ./clothing-small/* .

python3.8 ./datasets/create_tfrecords.py -type all -config configs/clothing-small-base.config -name CLOTHING_SMALL_BASE


python3.8 train_selected.py -config configs/clothing-small-base.config -name CLOTHING_SMALL_BASE -arch resnet -method sgd -mode train -save True