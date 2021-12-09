import torch
import time
import logging
import json
import pickle as pkl
from dataset import BatchData
from torch.utils.data import DataLoader
from model.transformer import Transformer
from torch.optim import Adam
from tqdm import tqdm

torch.manual_seed(1)
time_flag = time.strftime("%Y-%m-%d %H:%M:%S")

conf_p = "config/config.json"
with open(conf_p, "r") as f:
    conf = json.load(f)
print(conf)

## Config
# dataset
train_en_set_path = conf['train_en_set_path']
train_zh_set_path = conf['train_zh_set_path']
val_en_set_path = conf['val_en_set_path']
val_zh_set_path = conf['val_zh_set_path']
# save dir
checkpoints_dir = conf['checkpoints_dir']
log_dir = conf['log_dir']
model_path = conf['model_path']
sep_dir = conf['sep_dir']
ids_dir = conf['ids_dir']
vocab_dir = conf['vocab_dir']
# hyper-parameters
batch_size = conf['batch_size']
num_epoch = conf['num_epoch']
lr = conf['lr']
max_len = conf['max_len']
en_vocab_size = conf['en_vocab_size']
zh_vocab_size = conf['zh_vocab_size']
is_resume = bool(conf['is_resume'])
if is_resume:
    model_path = conf['model_path']

## Log
logging.basicConfig(level=logging.DEBUG,
                    filename=log_dir+'train_' + str(time_flag) + '_log.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
# read token2id
with open(vocab_dir+'token2id_en.json', 'r') as f:
    token2id_en = json.load(f)
with open(vocab_dir+'token2id_zh.json', 'r') as f:
    token2id_zh = json.load(f)
# read data
with open(ids_dir+'train_en_ids.pkl', 'rb') as f:
    train_en_ids = pkl.load(f)
with open(ids_dir+'train_zh_ids.pkl', 'rb') as f:
    train_zh_ids = pkl.load(f)
assert len(train_en_ids) == len(train_zh_ids), 'Train: The length of en and zh are not equal. '
with open(ids_dir+'val_en_ids.pkl', 'rb') as f:
    val_en_ids = pkl.load(f)
with open(ids_dir+'val_zh_ids.pkl', 'rb') as f:
    val_zh_ids = pkl.load(f)
assert len(val_en_ids) == len(val_zh_ids), 'Val: The length of en and zh are not equal. '

train_dataset = BatchData(train_en_ids, train_zh_ids, en_vocab_size, zh_vocab_size, max_len, token2id_en, token2id_zh)
val_dataset = BatchData(val_en_ids, val_zh_ids, en_vocab_size, zh_vocab_size, max_len, token2id_en, token2id_zh)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_laoder = DataLoader(val_dataset, batch_size=batch_size)

print('num train samples: ', len(train_en_ids))
print('num val samples: ', len(val_en_ids))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Transformer()
model.to(device)
print('device: ' + str(device))
optimizer = Adam(model.parameters(), lr=lr)

# train
model.train()
total_steps = len(train_loader)
for epoch in tqdm(range(num_epoch)):
    step = 0
    for i, batch_data in enumerate(train_loader):
        print('step: {} / {}'.format(step, total_steps))
        model.zero_grad()
        pred = model(batch_data)
        step += 1


