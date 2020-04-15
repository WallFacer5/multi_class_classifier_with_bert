# coding='utf-8'

import random
import argparse
import os


def trans_data(cates):
    if not os.path.exists(r'datasets/stanfordSentimentTreebank/sst_%s' % (str(cates))):
        os.mkdir(r'datasets/stanfordSentimentTreebank/sst_%s' % (str(cates)))

    train_f = open(r'datasets/stanfordSentimentTreebank/train.tsv', 'r')
    dev_f = open(r'datasets/stanfordSentimentTreebank/dev.tsv', 'r')
    test_f = open(r'datasets/stanfordSentimentTreebank/test.tsv', 'r')

    train_out = open(r'datasets/stanfordSentimentTreebank/sst_%s/train.tsv' % (str(cates)), 'w')
    dev_out = open(r'datasets/stanfordSentimentTreebank/sst_%s/dev.tsv' % (str(cates)), 'w')
    test_out = open(r'datasets/stanfordSentimentTreebank/sst_%s/test.tsv' % (str(cates)), 'w')

    write_data = []
    for line in train_f:
        line = line.strip('\n').split('\t')
        label1 = float(line[1]) * cates
        if label1 == int(label1):
            label = int(float(line[1]) * cates) - 1
        else:
            label = int(float(line[1]) * cates)
        if label >= cates:
            label = cates - 1
        line[1] = str(label)
        write_data.append('\t'.join(line))
    train_out.write('\n'.join(write_data))

    write_data = []
    for line in dev_f:
        line = line.strip('\n').split('\t')
        label = int(float(line[1]) * cates)
        if label >= cates:
            label = cates - 1
        line[1] = str(label)
        write_data.append('\t'.join(line))
    dev_out.write('\n'.join(write_data))

    write_data = []
    for line in test_f:
        line = line.strip('\n').split('\t')
        label = int(float(line[1]) * cates)
        if label >= cates:
            label = cates - 1
        line[1] = str(label)
        write_data.append('\t'.join(line))
    test_out.write('\n'.join(write_data))

    train_f.close()
    dev_f.close()
    test_f.close()

    train_out.close()
    dev_out.close()
    test_out.close()


def multi_sample(l, sample_num):
    ret_l = []
    while sample_num > len(l):
        ret_l += l
        sample_num -= len(l)
    return ret_l + random.sample(l, sample_num)


def rebalance(cates):
    if not os.path.exists(r'datasets/stanfordSentimentTreebank/sst_%s_rb' % (str(cates))):
        os.mkdir(r'datasets/stanfordSentimentTreebank/sst_%s_rb' % (str(cates)))

    train_f = open(r'datasets/stanfordSentimentTreebank/train.tsv', 'r')
    dev_f = open(r'datasets/stanfordSentimentTreebank/dev.tsv', 'r')
    test_f = open(r'datasets/stanfordSentimentTreebank/test.tsv', 'r')

    train_data = train_f.read()
    dev_data = dev_f.read()
    test_data = test_f.read()

    train_f.close()
    dev_f.close()
    test_f.close()

    train_data = train_data.split('\n')
    data_split = []
    for i in range(cates):
        data_split.append([])
    for i in range(len(train_data)):
        train_data[i] = train_data[i].split('\t')
        cur_label = int(float(train_data[i][1]) * cates)
        if cur_label >= cates:
            cur_label = cates - 1
        data_split[cur_label].append(train_data[i][0])
    data_count = list(map(lambda s: len(s), data_split))
    max_count = max(data_count)
    data_split = list(map(lambda l: l + multi_sample(l, max_count - len(l)), data_split))
    for i in range(cates):
        random.shuffle(data_split[i])
    train_data = []
    for i in range(max_count):
        for j in range(cates):
            cur_data = '\t'.join([str(data_split[j][i]), str(j)])
            train_data.append(cur_data)
    with open('datasets/stanfordSentimentTreebank/sst_%s_rb/train.tsv' % (str(cates)), 'w') as train_writer:
        train_writer.write('\n'.join(train_data))

    dev_data = dev_data.split('\n')
    data_split = []
    for i in range(cates):
        data_split.append([])
    for i in range(len(dev_data)):
        dev_data[i] = dev_data[i].split('\t')
        cur_label = int(float(dev_data[i][1]) * cates)
        if cur_label >= cates:
            cur_label = cates - 1
        data_split[cur_label].append(dev_data[i][0])
    data_count = list(map(lambda s: len(s), data_split))
    max_count = max(data_count)
    data_split = list(map(lambda l: l + multi_sample(l, max_count - len(l)), data_split))
    for i in range(cates):
        random.shuffle(data_split[i])
    dev_data = []
    for i in range(max_count):
        for j in range(cates):
            cur_data = '\t'.join([str(data_split[j][i]), str(j)])
            dev_data.append(cur_data)
    with open('datasets/stanfordSentimentTreebank/sst_%s_rb/dev.tsv' % (str(cates)), 'w') as dev_writer:
        dev_writer.write('\n'.join(dev_data))

    data_split = []
    with open('datasets/stanfordSentimentTreebank/sst_%s_rb/test.tsv' % (str(cates)), 'w') as test_out:
        test_data = test_data.split('\n')
        for line in test_data:
            line = line.strip('\n').split('\t')
            label = int(float(line[1]) * cates)
            if label >= cates:
                label = cates - 1
            line[1] = str(label)
            data_split.append('\t'.join(line))
        test_out.write('\n'.join(data_split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='oper', help='operation (rebalance/init)', type=str, required=True)
    parser.add_argument('-c', dest='cates', help='how many categories', type=str, required=True)
    args = parser.parse_args()
    print(args.oper)
    if args.oper == 'init':
        trans_data(int(args.cates))
    else:
        rebalance(int(args.cates))
