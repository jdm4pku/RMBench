#!/bin/bash

# # entity recognition
# DATA="data/split_v1"

# # Iterate over shots
# for SHOT in 4 8 16 32
# do
#     # Iterate over splits
#     for SPLIT in {1..7}
#     do
#          python src/llm/entity_llm.py \
#             --data $DATA \
#             --split $SPLIT \
#             --shot $SHOT \
#             --retrieve "similarity" \
#             --prompt "prompt/entity.txt" \
#             --model "gemma-2-9b" \
#             --moda "greedy" \
#             --output_dir "output/llm"
#     done
# done

# interaction extraction
DATA="data/split_v1"

# Iterate over shots
for SHOT in 4 8 16 32
do
    # Iterate over splits
    for SPLIT in {1..7}
    do
        CUDA_VISIBLE_DEVICES=4 python src/llm/interaction_llm.py \
            --data $DATA \
            --split $SPLIT \
            --shot $SHOT \
            --retrieve "similarity" \
            --prompt "prompt/interaction.txt" \
            --model "gemma-2-9b" \
            --moda "greedy" \
            --output_dir "output/llm-interaction"
    done
done
