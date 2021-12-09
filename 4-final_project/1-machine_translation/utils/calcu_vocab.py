import numpy as np
from tqdm import tqdm

train_en_set_path = 'data/en-zh/train.en'
train_zh_set_path = './data/en-zh/train.zh'

with open(train_en_set_path, 'r') as fe:
    en_lines = fe.readlines()
with open(train_zh_set_path, 'r', encoding='utf-8') as fz:
    zh_lines = fz.readlines()


en_len_list = []
zh_len_list = []
en_vocab = []
zh_vocab = []
for i in tqdm(range(len(en_lines))):
    en_len_list.append(len(en_lines[i].split(' ')))
    zh_len_list.append(len(zh_lines[i]))
    en_vocab.extend(en_lines[i].split(' '))
    zh_vocab.extend(list(zh_lines[i]))
    # print(list(zh_lines[i]))
    # exit()


en_len_list = np.array(en_len_list)
zh_len_list = np.array(zh_len_list)
print('max len en: ', max(en_len_list))
print('mean len en: ', en_len_list.mean())
print('max len zh: ', max(zh_len_list))
print('mean len zh: ', zh_len_list.mean())
print('en vocab size: ', len(list(set(en_vocab))))
print('zh vocab size: ', len(list(set(zh_vocab))))