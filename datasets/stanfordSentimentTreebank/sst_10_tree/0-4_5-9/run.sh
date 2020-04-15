#!/usr/bin/env bash

export BERT_BASE_DIR=/home/zyh/projects/multi_class_classifier_with_bert/py_dependency/uncased_L-12_H-768_A-12
export GLUE_DIR=/home/zyh/projects/multi_class_classifier_with_bert/datasets/stanfordSentimentTreebank/sst_10_tree/0-4_5-9

python py_dependency/bert/run_classifier.py \
  --task_name=SSTtree \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=10.0 \
  --output_dir=/home/zyh/output/sst_output_uncased_tree_10/0-4_5-9
        