# coding='utf-8'

import sys
import os
import pandas as pd
import random

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_PATH, '../'))
sys.path.append(ROOT_PATH)


class MultiClassifyTree(object):

    def __init__(self, data_dir, output_dir, rebalance, cates=5):
        self.classifiers = []
        self.cates = cates
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.rebalance = rebalance

        self.train_path = os.path.join(data_dir, 'train.tsv')
        self.dev_path = os.path.join(data_dir, 'dev.tsv')
        self.test_path = os.path.join(data_dir, 'test.tsv')

        self.train_s = []
        self.train_label = []

        self.dev_s = []
        self.dev_label = []

        self.test_s = []
        self.test_label = []

        self.data_distribution = []

        for i in range(cates):
            self.data_distribution.append(0)

        with open(self.train_path, 'r') as train_reader:
            for line in train_reader:
                line = line.strip('\n').split('\t')
                line[1] = int(line[1])
                self.train_s.append(line[0])
                self.train_label.append(line[1])
                self.data_distribution[line[1]] += 1

        with open(self.dev_path, 'r') as dev_reader:
            for line in dev_reader:
                line = line.strip('\n').split('\t')
                line[1] = int(line[1])
                self.dev_s.append(line[0])
                self.dev_label.append(line[1])

        with open(self.test_path, 'r') as test_reader:
            for line in test_reader:
                line = line.strip('\n').split('\t')
                line[1] = int(line[1])
                self.test_s.append(line[0])
                self.test_label.append(line[1])

        print('data_distribution: ', self.data_distribution)

    def split3(self, pool):
        total = sum(list(map(lambda n: self.data_distribution[n], pool)))
        avg_count = total / 3
        count1 = self.data_distribution[pool[0]]
        flag = 0
        while count1 < avg_count:
            flag += 1
            count1 = sum(self.data_distribution[pool[0]:pool[flag + 1]])
        flag12 = flag
        flag11 = flag - 1
        count2 = self.data_distribution[pool[-1]]
        flag = len(pool) - 1
        while count2 < avg_count:
            flag -= 1
            count2 = sum(self.data_distribution[pool[flag]:pool[-1] + 1])
        flag21 = flag
        flag22 = flag + 1

        if flag22 >= len(pool):
            flag22 = len(pool) - 1

        if flag11 < 0:
            flag11 = 0

        print(flag11, flag12, flag21, flag22)

        counts = []
        splits = []

        for i in range(4):
            counts.append([])
            for j in range(3):
                counts[i].append(0)

        splits.append([pool[:flag11 + 1], pool[flag11 + 1:flag21], pool[flag21:]])

        counts[0][0] = sum(self.data_distribution[pool[0]:pool[flag11 + 1]])
        counts[0][1] = sum(self.data_distribution[pool[flag11 + 1]:pool[flag21]])
        counts[0][2] = sum(self.data_distribution[pool[flag21]:pool[-1] + 1])

        splits.append([pool[:flag11 + 1], pool[flag11 + 1:flag22], pool[flag22:]])

        counts[1][0] = sum(self.data_distribution[pool[0]:pool[flag11 + 1]])
        counts[1][1] = sum(self.data_distribution[pool[flag11 + 1]:pool[flag22]])
        counts[1][2] = sum(self.data_distribution[pool[flag22]:pool[-1] + 1])

        splits.append([pool[:flag12 + 1], pool[flag12 + 1:flag21], pool[flag21:]])

        counts[2][0] = sum(self.data_distribution[pool[0]:pool[flag12 + 1]])
        counts[2][1] = sum(self.data_distribution[pool[flag12 + 1]:pool[flag21]])
        counts[2][2] = sum(self.data_distribution[pool[flag21]:pool[-1] + 1])

        splits.append([pool[:flag12 + 1], pool[flag12 + 1:flag22], pool[flag22:]])

        counts[3][0] = sum(self.data_distribution[pool[0]:pool[flag12 + 1]])
        counts[3][1] = sum(self.data_distribution[pool[flag12 + 1]:pool[flag22]])
        counts[3][2] = sum(self.data_distribution[pool[flag22]:pool[-1] + 1])

        dis = [max(counts[0]) - min(counts[0]),
               max(counts[1]) - min(counts[1]),
               max(counts[2]) - min(counts[2]),
               max(counts[3]) - min(counts[3])]

        max_index = dis.index(min(dis))
        return splits[max_index], dis[max_index]

    def split2(self, pool):
        min_dis = sum(self.data_distribution[pool[0]:pool[-1] + 1])
        flag = 0
        for i in range(1, len(pool)):
            cur_dis = abs(
                sum(self.data_distribution[pool[0]:pool[i]]) - sum(self.data_distribution[pool[i]:pool[-1] + 1]))
            if cur_dis < min_dis:
                min_dis = cur_dis
                flag = i
        return [pool[:flag], pool[flag:]], min_dis

    def split(self, pool, max_split_units=3):
        if len(pool) < 3:
            return [[i] for i in pool]
        splits3, dis3 = self.split3(pool)
        splits2, dis2 = self.split2(pool)
        print(dis2, dis3)
        if dis3 > dis2:
            return splits2
        else:
            return splits3

    def multi_sample(self, l, sample_num):
        ret_l = []
        while sample_num > len(l):
            ret_l += l
            sample_num -= len(l)
        return ret_l + random.sample(l, sample_num)

    def generate_single_classifier(self, split):
        single_path = '_'.join(list(map(lambda l: '-'.join([str(l[0]), str(l[-1])]), split)))
        data_dir = r'datasets/stanfordSentimentTreebank/sst_%s_tree' % (str(self.cates))
        single_path = os.path.join(data_dir, single_path)
        if not os.path.exists(single_path):
            os.mkdir(single_path)
        train_path = os.path.join(single_path, 'train.tsv')
        dev_path = os.path.join(single_path, 'dev.tsv')
        test_path = os.path.join(single_path, 'test.tsv')
        with open(train_path, 'w') as train_writer:
            data2write = []
            for i in range(len(self.train_label)):
                split_compare = list(map(lambda l: self.train_label[i] in l, split))
                if True in split_compare:
                    split_index = split_compare.index(True)
                    data2write.append(
                        '\t'.join(
                            [self.train_s[i], '-'.join([str(split[split_index][0]), str(split[split_index][-1])])]))
            train_writer.write('\n'.join(data2write))
        with open(dev_path, 'w') as dev_writer:
            data2write = []
            for i in range(len(self.dev_label)):
                split_compare = list(map(lambda l: self.dev_label[i] in l, split))
                if True in split_compare:
                    split_index = split_compare.index(True)
                    data2write.append(
                        '\t'.join(
                            [self.dev_s[i], '-'.join([str(split[split_index][0]), str(split[split_index][-1])])]))
            dev_writer.write('\n'.join(data2write))
        with open(test_path, 'w') as test_writer:
            data2write = []
            for i in range(len(self.test_label)):
                split_compare = list(map(lambda l: self.test_label[i] in l, split))
                if True in split_compare:
                    split_index = split_compare.index(True)
                    data2write.append(
                        '\t'.join(
                            [self.test_s[i], '-'.join([str(split[split_index][0]), str(split[split_index][-1])])]))
            test_writer.write('\n'.join(data2write))

        self.classifiers.append(single_path)

    def generate_single_classifier_rb(self, split):
        single_path = '_'.join(list(map(lambda l: '-'.join([str(l[0]), str(l[-1])]), split)))
        data_dir = r'datasets/stanfordSentimentTreebank/sst_%s_tree' % (str(self.cates))
        single_path = os.path.join(data_dir, single_path)
        if not os.path.exists(single_path):
            os.mkdir(single_path)
        train_path = os.path.join(single_path, 'train.tsv')
        dev_path = os.path.join(single_path, 'dev.tsv')
        test_path = os.path.join(single_path, 'test.tsv')
        with open(train_path, 'w') as train_writer:
            data2write = []
            rebalance_dict = {}
            for i in range(len(self.train_label)):
                split_compare = list(map(lambda l: self.train_label[i] in l, split))
                if True in split_compare:
                    split_index = split_compare.index(True)
                    data2write.append(
                        '\t'.join(
                            [self.train_s[i], '-'.join([str(split[split_index][0]), str(split[split_index][-1])])]))
                    cur_cate = '-'.join([str(split[split_index][0]), str(split[split_index][-1])])
                    if cur_cate not in rebalance_dict:
                        rebalance_dict[cur_cate] = {'len': 1, 'data': [self.train_s[i]]}
                    else:
                        rebalance_dict[cur_cate]['len'] += 1
                        rebalance_dict[cur_cate]['data'].append(self.train_s[i])
            max_len = 0
            for keys in rebalance_dict:
                if rebalance_dict[keys]['len'] > max_len:
                    max_len = rebalance_dict[keys]['len']

            for keys in rebalance_dict:
                data2write += list(
                    map(lambda s: '\t'.join([s, keys]),
                        self.multi_sample(rebalance_dict[keys]['data'], max_len - rebalance_dict[keys]['len'])))
            random.shuffle(data2write)

            train_writer.write('\n'.join(data2write))
        with open(dev_path, 'w') as dev_writer:
            data2write = []
            for i in range(len(self.dev_label)):
                split_compare = list(map(lambda l: self.dev_label[i] in l, split))
                if True in split_compare:
                    split_index = split_compare.index(True)
                    data2write.append(
                        '\t'.join(
                            [self.dev_s[i], '-'.join([str(split[split_index][0]), str(split[split_index][-1])])]))
            dev_writer.write('\n'.join(data2write))
        with open(test_path, 'w') as test_writer:
            data2write = []
            for i in range(len(self.test_label)):
                data2write.append('\t'.join([self.test_s[i], str(self.test_label[i])]))
            test_writer.write('\n'.join(data2write))

        self.classifiers.append(single_path)

    def generate_classifiers(self):
        if not os.path.exists(r'datasets/stanfordSentimentTreebank/sst_%s_tree' % (str(self.cates))):
            os.mkdir(r'datasets/stanfordSentimentTreebank/sst_%s_tree' % (str(self.cates)))
        pool = [[i for i in range(self.cates)]]
        while len(pool) < self.cates:
            next_pool = []
            for sub_pool in pool:
                split_result = self.split(sub_pool)
                print(split_result)
                if self.rebalance:
                    self.generate_single_classifier_rb(split_result)
                else:
                    self.generate_single_classifier(split_result)
                next_pool += split_result
            pool = next_pool
            print(pool)
        print(self.classifiers)

    def merge_results(self):
        results = []
        total = 0
        right = 0
        for i in range(len(self.test_label)):
            results.append([])
        for classifier in self.classifiers:
            with open(os.path.join(self.output_dir, classifier.split('/')[-1], 'test_compare.tsv'),
                      'r') as result_reader:
                i = -1
                for line in result_reader:
                    if i >= 0:
                        line = line.strip('\n').split('\t')
                        results[i].append(line[0])
                    i += 1
        self.confused_matrix = []
        result_writer = open(os.path.join(self.output_dir, 'test_result.tsv'), 'w')

        for i in range(self.cates):
            self.confused_matrix.append([])
            for j in range(self.cates):
                self.confused_matrix[i].append(0)
        for i in range(len(self.test_label)):
            cur_result = []
            print(results[i])
            for j in range(len(results[i])):
                interval = list(map(lambda s: int(s), results[i][j].split('-')))
                if not cur_result:
                    cur_result = interval
                else:
                    if interval[0] >= cur_result[0] and interval[1] <= cur_result[1]:
                        cur_result = interval
            if cur_result[0] != cur_result[1]:
                print('interval wrong!')
            cur_result = cur_result[0]
            results[i] = '\t'.join([self.test_s[i], str(cur_result)])

            index0 = self.test_label[i]
            index1 = cur_result

            if index0 == index1:
                right += 1
            total += 1

            self.confused_matrix[index0][index1] += 1
        result_writer.write('\n'.join(results))
        result_writer.close()

        with open(os.path.join(self.output_dir, 'test_accuracy.txt'), 'w') as acc_writer:
            acc = right / total
            acc_writer.write(
                '\n'.join(list(map(lambda l: '\t'.join(list(map(lambda s: str(s), l))), self.confused_matrix))))
            acc_writer.write('\n')
            acc_writer.write('Accuracy: ' + str(acc))

        data_df = pd.DataFrame(self.confused_matrix)
        data_df.columns = list(range(self.cates))
        data_df.index = list(range(self.cates))
        precision = []
        recall = []
        all_precision = 0
        for i in range(len(list(data_df.columns))):
            predict_sum = sum(data_df.values[:, i])
            actual_sum = sum(data_df.values[i, :])
            # 计算召回率
            if actual_sum == 0:
                recall.append(0)
            else:
                recall_value = data_df.values[i][i] / actual_sum
                recall.append(recall_value)
            # 计算准确率
            if predict_sum == 0:
                precision.append(0)
            else:
                precision_value = data_df.values[i][i] / predict_sum
                precision.append(precision_value)
            all_precision += data_df.values[i][i]
        # tf.logging.info('all classify right data count: {}'.format(all_precision))
        data_df.loc['准确率'] = precision
        recall.append("")
        data_df['召回率'] = recall

        output_cm_file = os.path.join(self.output_dir, 'test_confused_matrix.csv')
        data_df.to_csv(output_cm_file, index=True, header=True, sep=',')

    def test(self):
        self.generate_classifiers()


if __name__ == '__main__':
    tree = MultiClassifyTree('datasets/stanfordSentimentTreebank/sst_5', '/home/zyh/output/sst_output_cased_tree', 5)
    tree.test()
