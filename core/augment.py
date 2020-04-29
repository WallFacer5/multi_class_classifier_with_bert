# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from core.eda import *


# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha, num_aug=9):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').read().split('\n')
    if lines[-1] == '':
        lines.pop()

    for i, line in enumerate(lines):
        parts = line.split('\t')
        label = parts[-1]
        sentence = '\t'.join(parts[:-1])
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write('\t'.join([aug_sentence, label]) + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(
        num_aug))
