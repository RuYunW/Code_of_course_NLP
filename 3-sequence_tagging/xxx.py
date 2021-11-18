import pickle as pkl
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='log.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

with open('feature/train.pkl', 'rb') as f:
    file = pkl.load(f)
print(file[0]['text_info']['tens_words'].shape)  # torch.Size([64, 1024])
print(file[0]['text_info']['tens_sentence'].shape)  # torch.Size([1024])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        text_info = data['text_info']['tens_words'].to(self.device)
        return text_info

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1024, 7)

    def _feat_softmax(self, feat):
        feat_softmax = F.softmax(feat, dim=0)
        return feat_softmax

    def forward(self, input_data):
        output = self.fc(input_data)
        feat_softmax = self._feat_softmax(output)
        return feat_softmax

# input_data = file[0]['text_info']['tens_words'].to(device)
dataset = BatchData(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

model = Model()
model.to(device)

for d in dataloader:
    output = model(d)
    # print(output.shape)
    print(output.shape)  # torch.Size([5, 64, 7])
    logging.info({'shape': output.shape})
    exit()


