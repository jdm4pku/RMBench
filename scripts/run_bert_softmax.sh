export BERT_BASE_DIR=model/AI-ModelScope/bert-base-uncased

# Loop through splits 1 to 7
for i in {1..7}
do
  export DATA_DIR=data/split_v1/split_$i
  export OUTPUR_DIR=ckpts/bert_softmax/split_$i
  python src/bert/run_ner_softmax.py \
    --model_type=bert \
    --model_name_or_path=$BERT_BASE_DIR \
    --task_name=mner \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --loss_type=ce \
    --data_dir=$DATA_DIR \
    --train_max_seq_length=128 \
    --eval_max_seq_length=512 \
    --per_gpu_train_batch_size=24 \
    --per_gpu_eval_batch_size=24 \
    --learning_rate=3e-5 \
    --num_train_epochs=50 \
    --logging_steps=-1 \
    --save_steps=-1 \
    --output_dir=$OUTPUR_DIR \
    --overwrite_output_dir \
    --seed=42
done
