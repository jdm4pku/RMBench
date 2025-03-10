#!/bin/bash


# models=("qwen_similarity_1" "qwen_similarity_2" "llama_similarity_1" "llama_similarity_2" "deepseek_similarity_1" "deepseek_similarity_2")

# # 计算f1
# for model in "${models[@]}"
# do
#   # 遍历 1 到 12
#   for i in {1..7}
#   do
#     echo "Evaluating $model: python src/eval/compute_f1.py --split $i"
#     python src/eval/compute_f1.py \
#       --input_dir "metric/llm/$model"/predict \
#       --split $i \
#       --output_dir "metric/llm/$model"
#   done
# done

# for i in {1..5}
# do
#   python src/eval/compute_f1.py \
#     --input_dir "metric/llm/gemma_similarity_2/predict" \
#     --split $i \
#     --output_dir "metric/llm/gemma_similarity_2"
# done

# # python src/eval/compute_f1.py \
# #     --input_dir "metric/llm/llama_similarity_1/predict" \
# #     --split 5 \
# #     --output_dir "metric/llm/llama_similarity_1"
 
# python src/eval/compute_f1.py \
#     --input_dir "metric/llm/gemma_similarity_2/predict" \
#     --split 7 \
#     --output_dir "metric/llm/gemma_similarity_2"


# for i in {1..5}
# do
#   python src/eval/compute_f1.py \
#     --input_dir "metric/llm/gpt-4_similarity_2/predict" \
#     --split $i \
#     --output_dir "metric/llm/gpt-4_similarity_2"
# done

# python src/eval/compute_average.py \
#     --directory "metric/llm/gpt-4_similarity_2/f1"\


models=("deepseek" "gemma" "llama" "qwen")

for model in "${models[@]}"
do
  for i in 4 8 16
  do
    python src/eval/compute_rough.py \
      --input_dir "metric/llm/${model}_similarity_${i}/predict" \
      --split 1 \
      --output_dir "metric/llm/${model}_similarity_${i}"
  done
done

# for i in {1..5}
# do
#   python src/eval/compute_rough.py \
#     --input_dir "metric/llm/qwen_similarity_2/predict" \
#     --split $i \
#     --output_dir "metric/llm/qwen_similarity_2"
# done

# python src/eval/compute_rough.py \
#     --input_dir "metric/llm/llama_similarity_2/predict" \
#     --split 5 \
#     --output_dir "metric/llm/llama_similarity_2"

# python src/eval/compute_rough.py \
#     --input_dir "metric/llm/qwen_similarity_2/predict" \
#     --split 7 \
#     --output_dir "metric/llm/qwen_similarity_2"

# for i in {1..4}
# do
#   python src/eval/compute_rough.py \
#     --input_dir "metric/llm/gpt-4_similarity_2/predict" \
#     --split $i \
#     --output_dir "metric/llm/gpt-4_similarity_2"
# done

# python src/eval/compute_average.py \
#     --directory "metric/llm/gpt-4_similarity_2/rough"\

