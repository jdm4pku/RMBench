#!/bin/bash

# DATA="data/split_v1"

# # Iterate over shots
# for SHOT in 4 8 16 32
# do
#     # Iterate over splits
#     for SPLIT in {1..7}
#     do
#         CUDA_VISIBLE_DEVICES=6 python src/llm/entity_llm.py \
#             --data $DATA \
#             --split $SPLIT \
#             --shot $SHOT \
#             --retrieve "similarity" \
#             --prompt "prompt/entity.txt" \
#             --model "llama3.1-8b" \
#             --moda "greedy" \
#             --output_dir "output/llm"
#     done
# done

# interaction extraction
DATA="data/split_v1"

# Iterate over shots
for SHOT in 1 2 4 8 16 32
do
    # Iterate over splits
    for SPLIT in {1..7}
    do
        CUDA_VISIBLE_DEVICES=2 python src/llm/interaction_llm.py \
            --data $DATA \
            --split $SPLIT \
            --shot $SHOT \
            --retrieve "similarity" \
            --prompt "prompt/interaction.txt" \
            --model "llama3.1-8b" \
            --moda "greedy" \
            --output_dir "output/llm-interaction"
    done
done
