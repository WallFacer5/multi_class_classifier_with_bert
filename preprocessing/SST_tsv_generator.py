sentence_path = r'/Users/zhangyanhan/Downloads/stanfordSentimentTreebank/datasetSentences.txt'
split_path = r'/Users/zhangyanhan/Downloads/stanfordSentimentTreebank/datasetSplit.txt'
label_path = r'/Users/zhangyanhan/Downloads/stanfordSentimentTreebank/sentiment_labels.txt'

sentence_file = open(sentence_path, 'r')
split_file = open(split_path, 'r')
label_file = open(label_path, 'r')

sentences = sentence_file.read()
splits = split_file.read()
labels = split_file.read()

sentence_file.close()
split_file.close()
label_file.close()

sentences = sentences.split('\n')[1:-1]
splits = splits.split('\n')[1:-1]
labels = labels.split('\n')[1:-1]

train_data = []
dev_data = []
test_data = []

for i in range(len(sentences)):
    