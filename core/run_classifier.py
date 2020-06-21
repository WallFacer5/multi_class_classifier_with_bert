# coding='utf-8'

import sys
import os
import argparse

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_PATH, '../'))
sys.path.append(ROOT_PATH)

from core.multi_classify_tree import MultiClassifyTree
from core.bert_tree import BertTree


def run_classifier(backend, data_dir, output_dir, rebalance, eda, cates=5):
    if backend.lower().split('_')[0]=='bert':
        model = BertTree(data_dir, output_dir, rebalance, eda, cates)
        case = backend.lower().split('_')[1]
        size = backend.lower().split('_')[2]
        model.case_setter(case)
        model.size_setter(size)
        model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', dest='backend', help='backend model type', type=str, required=True)
    parser.add_argument('-d', dest='data_dir', help='data directory', type=str, required=True)
    parser.add_argument('-o', dest='output_dir', help='output directory', type=str, required=True)
    parser.add_argument('-c', dest='cates', help='categories', type=int, required=True)
    parser.add_argument('-r', dest='rebalance', help='do rebalance or not', type=bool, required=True)
    parser.add_argument('-eda', dest='eda', help='do eda or not', type=bool, required=False, default=False)
    args = parser.parse_args()
    run_classifier(args.backend, args.data_dir, args.output_dir, args.rebalance, args.eda, args.cates)
