sentence_path = r'../datasets/stanfordSentimentTreebank/datasetSentences.txt'
split_path = r'../datasets/stanfordSentimentTreebank/datasetSplit.txt'
label_path = r'../datasets/stanfordSentimentTreebank/sentiment_labels.txt'
sentence_dict_path = r'../datasets/stanfordSentimentTreebank/dictionary.txt'

sentence_file = open(sentence_path, 'r')
split_file = open(split_path, 'r')
label_file = open(label_path, 'r')
sentence_dict_file = open(sentence_dict_path, 'r')

sentences = sentence_file.read()
splits = split_file.read()
labels = label_file.read()
sentence_dict = sentence_dict_file.read()

sentence_file.close()
split_file.close()
label_file.close()
sentence_dict_file.close()

sentences = sentences.split('\n')[1:-1]
splits = splits.split('\n')[1:-1]
labels = labels.split('\n')[1:-1]
sentence_dict = sentence_dict.split('\n')[:-1]

train_data = []
dev_data = []
test_data = []

for i in range(len(labels)):
    labels[i] = labels[i].split('|')
    if i == int(labels[i][0]):
        labels[i] = float(labels[i][1])
    else:
        print('loss: %d' % i)

phrase_dict = {}

for i in range(len(sentence_dict)):
    sentence_dict[i] = sentence_dict[i].split('|')
    phrase_dict[sentence_dict[i][0]] = labels[int(sentence_dict[i][1])]

distribution = [0, 0, 0, 0, 0]

for i in range(len(sentences)):
    sentences[i] = sentences[i].split('\t')
    splits[i] = splits[i].split(',')
    if i + 1 == int(sentences[i][0]) and i + 1 == int(splits[i][0]):
        tmp = [sentences[i][1]]
        if tmp[0] in phrase_dict:
            tmp.append(phrase_dict[tmp[0]])

            if tmp[1] <= 0.2:
                distribution[0] += 1
            elif tmp[1] <= 0.4:
                distribution[1] += 1
            elif tmp[1] <= 0.6:
                distribution[2] += 1
            elif tmp[1] <= 0.8:
                distribution[3] += 1
            else:
                distribution[4] += 1

            tmp[1] = str(tmp[1])

            if int(splits[i][1]) == 1:
                train_data.append('\t'.join(tmp))
            elif int(splits[i][1]) == 2:
                test_data.append('\t'.join(tmp))
            else:
                dev_data.append('\t'.join(tmp))
        else:
            print('loss sentence %d %s' % (i, tmp[0]))

train_file = open(r'../datasets/stanfordSentimentTreebank/train.tsv', 'w')
test_file = open(r'../datasets/stanfordSentimentTreebank/test.tsv', 'w')
dev_file = open(r'../datasets/stanfordSentimentTreebank/dev.tsv', 'w')

train_file.write('\n'.join(train_data))
test_file.write('\n'.join(test_data))
dev_file.write('\n'.join(dev_data))

train_file.close()
test_file.close()
dev_file.close()
