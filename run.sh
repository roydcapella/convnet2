#!/bin/bash
#mv train_sample.txt train.txt
#(venv) panchito2@iafconnectvm4:/data/practica/otro/convnet2$ mv test_sample.txt test.txt
#(venv) panchito2@iafconnectvm4:/data/practica/otro/convnet2$ pyth

#wget https://www.dropbox.com/s/c3h9e0o6ajydbao/clothing-small.zip

#unzip -uq 'clothing-small.zip' -d '.'
#rm clothing-small.zip
#mv ./clothing-small/* .

python ./datasets/create_tfrecords.py -type all -config configs/clothing-small-base.config -name CLOTHING_SMALL_BASE


python train_selected.py -config configs/clothing-small-base.config -name CLOTHING_SMALL_SGD -arch resnet -method sgd -mode train -save True
python train_selected.py -config configs/clothing-small-base.config -name CLOTHING_SMALL_ADAM -arch resnet -method adam -mode train -save True

python train_selected.py -config configs/clothing-small-base.config -name CLOTHING_SMALL_SGD -arch alexnet -method sgd -mode train -save True
python train_selected.py -config configs/clothing-small-base.config -name CLOTHING_SMALL_ADAM -arch alexnet -method adam -mode train -save True
