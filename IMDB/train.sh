#!/bin/bash -l


for((k=1;k<=100;k++))
do
python -u IMDB_model.py -num ${k}
done
