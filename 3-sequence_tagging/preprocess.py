import torch
import argparse
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser(description='提取文字，使用chinese-roberta-wwm-ext-large')
parser.add_argument('-d', default='train', type=str, help='dataset type [train, test]')
args = parser.parse_args()

text_path = 'data/' + args.d + '_corpus.txt'
label_path = 'data/' + args.d + '_label.txt'
save_path = 'feature/' + args.d + '.pkl'

MAX_LENGTH = 64
BATCH_SIZE = 64

class BatchData(Dataset):
    def __init__(self, text_path, label_path, max_len, text_lines, label_lines):
        self.text_path = text_path
        self.label_path = label_path
        self.textlines = text_lines
        self.labellines = label_lines
        self.tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.textlines)

    def __getitem__(self, item):
        text = ''.join(self.textlines[item].split(' ')[:-1])
        tag = [self.tag2id[tag] for tag in self.labellines[item].split(' ')[:-1]]
        sentence_len = len(tag)
        if len(tag) < self.max_len:
            tag = tag + [self.tag2id['PAD'] for p in range(self.max_len - len(tag))]
        else:
            tag = tag[:self.max_len]
        tag = torch.tensor(tag).to(device)

        text_info = self.tokenizer.encode_plus(text, max_length=self.max_len,
                                               padding='max_length', truncation=True)
        text_info['input_ids'] = torch.tensor(text_info['input_ids']).to(self.device)
        text_info['token_type_ids'] = torch.tensor(text_info['token_type_ids']).to(self.device)
        text_info['attention_mask'] = torch.tensor(text_info['attention_mask']).to(self.device)
        text_info['val_token_type_ids'] = torch.ones(self.max_len)*0
        text_info['val_token_type_ids'] = torch.tensor(text_info['val_token_type_ids']).to(self.device)

        return {'text_info': text_info, 'tag': tag, 'sentence_len': sentence_len}


# data_list = file_reader(text_path, label_path)
text_lines = open(text_path).readlines()
label_lines = open(label_path).readlines()
dataset = BatchData(text_path, label_path, MAX_LENGTH, text_lines, label_lines)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
model.eval()

counter = 0
sample_list = []
print('feature extracting...')
with torch.no_grad():
    for i, data_info in tqdm(enumerate(dataloader)):
        text_info = data_info['text_info']
        tag_info = data_info['tag']
        sentence_len = data_info['sentence_len']
        text_feature = model(text_info['input_ids'], token_type_ids=text_info['token_type_ids'],
                             attention_mask=text_info['attention_mask'])
        for j in range(len(tag_info)):
            tag = tag_info[j].to('cpu')
            text_input_ids = text_info['input_ids'][j].to('cpu')
            text_token_type_ids = text_info['token_type_ids'][j].to('cpu')
            text_attention_mask = text_info['attention_mask'][j].to('cpu')
            text_val_token_type_ids = text_info['val_token_type_ids'][j].to('cpu')

            text_feature_words = text_feature[0][j].to('cpu')
            text_feature_sentence = text_feature[1][j].to('cpu')

            sample = {'tags': tag,
                      'sentence_len': sentence_len,
                      'text_info': {
                          'input_ids': text_input_ids,
                          'token_type_ids': text_token_type_ids,
                          'attention_mask': text_attention_mask,
                          'val_token_type_ids': text_val_token_type_ids,
                          'tens_words': text_feature_words,
                          'tens_sentence': text_feature_sentence
                      }}
            sample_list.append(sample)
            # counter += 1
print(len(sample_list))

print('successfully extract feature')
with open(save_path, 'wb') as fw:
    pkl.dump(sample_list, fw)
print('save file ')
