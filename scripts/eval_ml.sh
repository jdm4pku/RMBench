#!/bin/bash

# # 定义模型类型的数组
# models=("hmm" "crf" "bilstm")

# # 计算f1
# for model in "${models[@]}"
# do
#   # 遍历 1 到 12
#   for i in {1..7}
#   do
#     echo "Evaluating $model: python src/eval/compute_f1.py --split $i"
#     python src/eval/compute_f1.py \
#       --input_dir "output/ml/$model" \
#       --split $i \
#       --output_dir "metric/ml/$model"
#   done
# done

# 计算rough

for i in {1..5}
do
  python src/eval/compute_rough.py \
    --input_dir "metric/ml/hmm/predict" \
    --split $i \
    --output_dir "metric/ml/hmm"
done

python src/eval/compute_rough.py \
    --input_dir "metric/ml/hmm/predict" \
    --split 7 \
    --output_dir "metric/ml/hmm"
  
python src/eval/compute_average.py \
    --directory "metric/ml/hmm/rough"\