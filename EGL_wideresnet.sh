#!/bin/bash -l


for((i=0;i<25;i++));
do
python -u EGL_wideresnet.py
python -u EGL_wideresnet_train.py
done
