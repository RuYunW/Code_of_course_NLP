from tqdm import tqdm
import re
import jieba
import json
import pickle as pkl
import logging

conf_p = "config/config.json"
with open(conf_p, "r") as f:
    conf = json.load(f)

## Config
train_en_file_path = conf['train_en_set_path']
val_en_file_path = conf['val_en_set_path']
test_en_file_path = conf['test_en_set_path']
train_zh_file_path = conf['train_zh_set_path']
val_zh_file_path = conf['val_zh_set_path']
test_zh_file_path = conf['test_zh_set_path']
features_dir = conf['features_dir']
vocab_dir = conf['vocab_dir']
sep_dir = conf['sep_dir']
ids_dir = conf['ids_dir']

en_vocab_size = conf['en_vocab_size']
zh_vocab_size = conf['zh_vocab_size']

## Log
logging.basicConfig(level=logging.DEBUG,
                    filename=features_dir + 'vocab_log.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

## Global Parameters
en_vocab = {}
zh_vocab = {}


def not_none(token):
    return token != ' ' or token != ''


def clean(text):
    # remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    remove_chars = '[’"#$%&\'()*+-/<=>@★、…【】《》“”‘’[\\]^_`{|}~]+'
    text = re.sub(r'[{}]+'.format(remove_chars), ' ', text)
    text = text.lower()
    return text
    # return re.sub(remove_chars, ' ', text)


def is_not_space(text):
    return text != ' '


def remove_none(text):
    new_list = text.split(' ')
    new_list_2 = list(filter(None, new_list))
    text_2 = ' '.join(new_list_2)
    return text_2


def en_sep(en_file_path):
    en_text_list = []
    with open(en_file_path, 'r', encoding='utf-8') as fe:
        en_lines = fe.readlines()
    for line in tqdm(en_lines):
        line = line.replace('\n', '')
        en_text = clean(line)
        en_text = remove_none(en_text)
        en_text_list.append(en_text)
        for token in line.split(' '):
            if token in en_vocab:
                en_vocab[token] += 1
            else:
                en_vocab[token] = 1
    return en_text_list


def zh_sep(zh_file_path):
    zh_text_list = []
    with open(zh_file_path, 'r', encoding='utf-8') as fz:
        zh_lines = fz.readlines()
    for line in tqdm(zh_lines):
        line = line.replace('\n', '')
        zh_text = clean(line)
        seg_list = list(jieba.cut(zh_text, cut_all=False))
        seg_list = list(filter(is_not_space, seg_list))
        zh_text_list.append(' '.join(seg_list))
        for token in seg_list:
            if token in zh_vocab:
                zh_vocab[token] += 1
            else:
                zh_vocab[token] = 1
    return zh_text_list


def sort_dict(vocab_dict: dict, vocab_size: int):
    sorted_vocab_temp = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab = {}
    vocab_keys = sorted_vocab_temp[:vocab_size]
    token2id = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
    id2token = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>', 3: '<UNK>'}
    for key, value in tqdm(vocab_keys):
        vocab[key] = value
        token2id[key] = len(token2id)
        id2token[len(id2token)] = key
    return vocab, token2id, id2token


def token_indexing(text_list, token2id):
    ids_list = []
    for sentence in tqdm(text_list):
        sen_ids = []
        for token in sentence.split(' '):
            if token in token2id:
                sen_ids.append(token2id[token])
            else:
                sen_ids.append(token2id['<UNK>'])
        ids_list.append(sen_ids)
    return ids_list


def save_txt(text_data, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as f:
        for text in tqdm(text_data):
            f.write(text + '\n')


def save_json(json_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)


def save_ids(ids_list, save_path):
    with open(save_path, 'wb') as f:
        pkl.dump(ids_list, f)


## Data Clean
# English
print('English data cleaning...')
train_en_text_list = en_sep(train_en_file_path)
val_en_text_list = en_sep(val_en_file_path)
test_en_text_list = en_sep(test_en_file_path)
# Chinese
print('Chinese data cleaning...')
train_zh_text_list = zh_sep(train_zh_file_path)
val_zh_text_list = zh_sep(val_zh_file_path)
test_zh_text_list = zh_sep(test_zh_file_path)

logging.info({'original en vocab size: {}, original zh vocab size: {}'.format(len(en_vocab), len(zh_vocab))})
## Vocab
en_vocab, token2id_en, id2token_en = sort_dict(en_vocab, en_vocab_size)
zh_vocab, token2id_zh, id2token_zh = sort_dict(zh_vocab, zh_vocab_size)


## Token indexing
print('Token indexing...')
# English
train_en_ids_list = token_indexing(train_en_text_list, token2id_en)
val_en_ids_list = token_indexing(val_en_text_list, token2id_en)
test_en_ids_list = token_indexing(test_en_text_list, token2id_en)

# print(test_en_ids_list)
# exit()
# Chinese
train_zh_ids_list = token_indexing(train_zh_text_list, token2id_zh)
val_zh_ids_list = token_indexing(val_zh_text_list, token2id_zh)
test_zh_ids_list = token_indexing(test_zh_text_list, token2id_zh)

assert len(train_en_text_list) == len(train_zh_text_list) and len(train_en_ids_list) == len(
    train_zh_ids_list), 'The length of train data is not matched!'
assert len(val_en_text_list) == len(val_zh_text_list) and len(val_en_ids_list) == len(
    val_zh_ids_list), 'The length of val data is not matched!'
assert len(test_en_text_list) == len(test_zh_text_list) and len(test_en_ids_list) == len(
    test_zh_ids_list), 'The length of test data is not matched!'

## Data Save
print('Data saving...')
# english sep
save_txt(train_en_text_list, sep_dir + 'train_en.txt')
save_txt(val_en_text_list, sep_dir + 'val_en.txt')
save_txt(test_en_text_list, sep_dir + 'test_en.txt')
# english vocab
save_json(en_vocab, vocab_dir + 'en_vocab.json')
save_json(token2id_en, vocab_dir + 'token2id_en.json')
save_json(id2token_en, vocab_dir + 'id2token_en.json')
# chinese sep
save_txt(train_zh_text_list, sep_dir + 'train_zh.txt')
save_txt(val_zh_text_list, sep_dir + 'val_zh.txt')
save_txt(test_zh_text_list, sep_dir + 'test_zh.txt')
# chinese vocab
save_json(zh_vocab, vocab_dir + 'zh_vocab.json')
save_json(token2id_zh, vocab_dir + 'token2id_zh.json')
save_json(id2token_zh, vocab_dir + 'id2token_zh.json')
# english ids
save_ids(train_en_ids_list, ids_dir + 'train_en_ids.pkl')
save_ids(val_en_ids_list, ids_dir + 'val_en_ids.pkl')
save_ids(test_en_ids_list, ids_dir + 'test_en_ids.pkl')
# chinese ids
save_ids(train_zh_ids_list, ids_dir + 'train_zh_ids.pkl')
save_ids(val_zh_ids_list, ids_dir + 'val_zh_ids.pkl')
save_ids(test_zh_ids_list, ids_dir + 'test_zh_ids.pkl')

logging.info(
    'en_vocab_size: {}|    zh_vocab_size: {}|    num_train_samples: {}|    num_val_samples: {}|    num_test_samples: {}'.format(
        len(en_vocab), len(zh_vocab), len(train_en_ids_list), len(val_en_ids_list), len(test_en_ids_list)))
print('Data has been saved.')
