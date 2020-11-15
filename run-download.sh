#!/bin/bash
echo "downloading dataset... "
wget https://www.dropbox.com/s/c3h9e0o6ajydbao/clothing-small.zip
mv 'clothing-small.zip' ./data/
unzip -uq 'data/clothing-small.zip' -d 'data/'
rm "data/clothing-small.zip"
mv 'data/test_sample.txt' 'data/test.txt'
mv 'data/train_sample.txt' 'data/train.txt'