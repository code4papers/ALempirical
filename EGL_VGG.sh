#!/bin/bash -l

for((i=0;i<25;i++));
do
python -u EGL_VGG.py
python -u EGL_VGG_train.py
done
