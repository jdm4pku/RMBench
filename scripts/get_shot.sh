#!/bin/bash

# 设置默认参数
DATA="data/split_v1"
SHOT=32

# 遍历 split 从 1 到 7
for SPLIT in {1..7}
do
    # 运行 Python 脚本
    python src/llm/get_shot.py \
        --data $DATA \
        --split $SPLIT \
        --shot $SHOT
done
