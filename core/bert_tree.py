# coding='utf-8'

import sys
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_PATH, '../'))
sys.path.append(ROOT_PATH)

from core.multi_classify_tree import MultiClassifyTree


class BertTree(MultiClassifyTree):
    def case_setter(self, case):
        self.case = case

    def size_setter(self, size):
        self.size = size

    def generate_run_bash(self, classifier):
        print('generating bash', classifier)
        bash_path = os.path.join(classifier, 'run.sh')
        run_path = 'py_dependency/bert/run_classifier.py'
        out_dir = os.path.join(self.output_dir, classifier.split('/')[-1])
        if self.case == 'case':
            case = ''
            do_lower = 'False'
        else:
            case = 'un'
            do_lower = 'True'
        if self.size == 'large':
            bert_name = 'wwm_{case}cased_L-24_H-1024_A-16'.format(case=case)
        else:
            bert_name = '{case}cased_L-12_H-768_A-12'.format(case=case)
        run_sh = '''#!/usr/bin/env bash

export BERT_BASE_DIR=~/multi_class_classifier_with_bert/py_dependency/{bert_name}
export GLUE_DIR=~/multi_class_classifier_with_bert/{classifier}

python {run_path} \\
  --task_name=SSTtree \\
  --do_train=true \\
  --do_eval=true \\
  --do_predict=true \\
  --data_dir=$GLUE_DIR \\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \\
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \\
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \\
  --do_lower_case={do_lower} \\
  --max_seq_length=64 \\
  --train_batch_size=64 \\
  --learning_rate=1e-5 \\
  --num_train_epochs=20.0 \\
  --save_checkpoints_steps=10000\\
  --output_dir={out_dir}
        '''.format(classifier=classifier, run_path=run_path, out_dir=out_dir, bert_name=bert_name, do_lower=do_lower)

        with open(bash_path, 'w') as bash_writer:
            bash_writer.write(run_sh)

        bash_log = os.popen('bash {bash_path}'.format(bash_path=bash_path))

        print(bash_log.read())

    def run_classification(self):
        for classifier in self.classifiers:
            self.generate_run_bash(classifier)

    def test(self):
        self.generate_classifiers()
        self.run_classification()
        self.merge_results()
        self.merge_results_old()


if __name__ == '__main__':
    tree = BertTree('datasets/stanfordSentimentTreebank/sst_5', '/home/zyh/output/sst_output_cased_tree', 5)
    tree.test()
