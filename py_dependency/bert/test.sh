#!/usr/bin/env bash

export BERT_BASE_DIR=/home/zyh/projects/cased_L-12_H-768_A-12
export GLUE_DIR=/home/zyh/projects/multi_class_classifier_with_bert/datasets/stanfordSentimentTreebank
export TRAINED_CLASSIFIER=/home/zyh/output/sst_output

CUDA_VISIBLE_DEVICES=1 python run_classifier.py \
  --task_name=SST \
  --do_predict=true \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/ \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/test_output/