#!/bin/bash -l

for((i=0;i<25;i++));
do
python -u EGL_VGG19.py
python -u EGL_VGG19_train.py
done
