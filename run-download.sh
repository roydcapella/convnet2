#!/bin/bash
echo "downloading dataset... "
wget https://www.dropbox.com/s/c3h9e0o6ajydbao/clothing-small.zip

unzip -uq 'clothing-small.zip' -d '.'
rm clothing-small.zip
mv ./clothing-small/* .
mv test_sample.txt test.txt
mv train_sample.txt train.txt
