#!/bin/bash

# 遍历 1 到 12
for i in {1..7}
do
  echo "Running: python src/ml/entity_sm.py --split $i"
  python src/ml/entity_sm.py --split $i
done