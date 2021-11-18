# -*- coding: utf-8 -*-
import os
import json
import torch
import argparse
import numpy as np
import pickle as pk
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser(description='提取文字，使用chinese-roberta-wwm-ext-large')
# parser.add_argument('-l', "--label_path", help='输入label所在目录')
parser.add_argument('-c', "--corpus_path", help='输入corpus所在目录')
# parser.add_argument('-s', "--save_path", help='输入保存目录')
args = parser.parse_args()

# assert os.path.exists(args.label_path), 'Label path not exist, please check!'
assert os.path.exists(args.corpus_path), 'Corpus file not exist, please check!'
# assert os.path.exists(args.save_path), 'Save path not exist, please check!'

# label_path = args.label_path
corpus_path = args.corpus_path
# save_dir = args.save_path

BATCH_SIZE = 64
TEXT_MAX_LEN = 32



#
# with open(label_path, 'r', encoding='utf-8') as f:
#     label_data = json.load(f)['data']
# # with open(orpus_path, 'r', encoding='utf-8') as ff:
# #     text2id = json.load(ff)
#
# all_data = []
# for sample_id, sample_info in label_data.items():
#     for t in sample_info['texts']:
#         all_data.append({'sample_id': sample_id,
#                          'roberta_text': t['roberta_text'],
#                          'text_id': text2id[sample_id][t['roberta_text']]})
#
#
# class BatchData(Dataset):
#     def __init__(self, all_data, text_max_len=TEXT_MAX_LEN):
#         self.all_data = all_data
#         self.type_dic = {'roberta_text': 0, 'resnext_roi': 1, 'resnext_img': 2}
#         self.text_max_len = text_max_len
#         self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     def __len__(self):
#         return len(self.all_data)
#
#     def __getitem__(self, idx):
#         data = self.all_data[idx]
#         sample_id = data['sample_id']
#         text = data['roberta_text']
#         text_id = data['text_id']
#
#         text_info = self.tokenizer.encode_plus(text, max_length=self.text_max_len,
#                                                padding='nax_length', truncation=True)
#         text_info['input_ids'] = torch.tensor(text_info['input_ids']).to(self.device)
#         text_info['token_type_ids'] = torch.tensor(text_info['token_type_ids']).to(self.device)
#         text_info['attention_mask'] = torch.tensor(text_info['attention_mask']).to(self.device)
#         text_info['val_token_type_ids'] = torch.ones(self.text_max_len) * self.type_dic['roberta_text']
#         text_info['val_token_type_ids'] = torch.tensor(text_info['val_token_type_ids']).to(self.device)
#
#         return {'sample_id': sample_id,
#                 'roberta_text': text,
#                 'text_id': text_id,
#                 'text_info': text_info}
#
#
# dataset = BatchData(all_data)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
# model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# model = model.to(device)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
# model.eval()
#
# with torch.no_grad():
#     for i, text_info in tqdm(enumerate(dataloader)):
#         sample_id = text_info['sample_id']
#         text_info = text_info['text_info']
#         text_feature = model(text_info['input_ids'], token_type_ids=text_info['token_type_ids'],
#                              attention_mask=text_info['attention_mask'])
#
#         for j in range(len(sample_id)):
#             save_sample_id = text_info['sample_id'][j]
#             text_id = text_info['text_id'][j]
#
#             text_input_ids = text_info['input_ids'][j].to('cpu')
#             text_token_type_ids = text_info['token_type_ids'][j].to('cpu')
#             text_attention_mask = text_info['attention_mask'][j].to('cpu')
#             text_val_token_type_ids = text_info['val_token_type_ids'][j].to('cpu')
#
#             text_feature_words = text_feature[0][j].to('cpu')
#             text_feature_sentence = text_feature[1][j].to('cpu')
#
#             sample = {'sample_id': save_sample_id,
#                       'text_id': text_id,
#                       'text_info': {
#                           'input_ids': text_input_ids,
#                           'token_type_ids': text_token_type_ids,
#                           'attention_mask': text_attention_mask,
#                           'val_token_type_ids': text_val_token_type_ids,
#                           'tens_words': text_feature_words,
#                           'tens_sentence': text_feature_sentence
#                       }}
#
#             with open(save_dir + save_sample_id + '/' + text_id + '.pkl', 'wb') as f:
#                 pk.dump(sample, f)
