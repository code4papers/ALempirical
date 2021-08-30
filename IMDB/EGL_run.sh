#!/bin/bash -l

for((i=0;i<100;i++));
do
python -u EGL_IMDB.py
python -u EGL_IMDB_train.py
done
