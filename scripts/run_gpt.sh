DATA="data/split_v1"

# Iterate over shots
for SHOT in 4 8 16 32
do
    # Iterate over splits
    # for SPLIT in {4..7}
    for SPLIT in 1
    do
        CUDA_VISIBLE_DEVICES=6 python src/llm/entity_gpt.py \
            --data $DATA \
            --split $SPLIT \
            --shot $SHOT \
            --retrieve "similarity" \
            --prompt "prompt/entity.txt" \
            --model "gpt-4" \
            --moda "greedy" \
            --output_dir "output/llm"
    done
done

# # interaction extraction
# DATA="data/split_v1"

# # Iterate over shots
# for SHOT in 1 2
# do
#     # Iterate over splits
#     for SPLIT in 1 2 3 4
#     do
#         CUDA_VISIBLE_DEVICES=2 python src/llm/interaction_gpt.py \
#             --data $DATA \
#             --split $SPLIT \
#             --shot $SHOT \
#             --retrieve "similarity" \
#             --prompt "prompt/interaction.txt" \
#             --model "gpt-4" \
#             --moda "greedy" \
#             --output_dir "output/llm-interaction"
#     done
# done