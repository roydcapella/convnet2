#!/bin/bash
echo "Train RESNET SGD"
python3.8 train_selected.py -config configs/clothing-small-sgd.config -name CLOTHING_SMALL_SGD -arch resnet -method sgd -mode train -save True

echo "Train RESNET ADAM"
#python3.8 train_selected.py -config configs/clothing-small-adam.config -name CLOTHING_SMALL_ADAM -arch resnet -method adam -mode train -save True

echo "ALEXNET SGD"
python3.8 train_selected.py -config configs/clothing-small-sgd.config -name CLOTHING_SMALL_SGD -arch alexnet -method sgd -mode train -save True

echo "ALEXNET ADAM"
#python3.8 train_selected.py -config configs/clothing-small-adam.config -name CLOTHING_SMALL_ADAM -arch alexnet -method adam -mode train -save True
