#!/bin/bash -l


for((i=0;i<25;i++));
do
python -u NiN_egl.py
python -u NIN_egl_train.py
done
