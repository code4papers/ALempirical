#!/bin/bash -l


for((k=1;k<=100;k++))
do
python -u Lenet_5.py -num ${k}
done
