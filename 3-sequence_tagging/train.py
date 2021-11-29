import math
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import pickle as pkl
from tqdm import tqdm
import torch.optim as optim
from dataset import BatchData
from utils import pkl_reader, dev, create_dir_not_exist
from model.bert_lm import BertLM
import logging

import time
import json

torch.manual_seed(1)
time_flag = time.strftime("%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='log/train_' + str(time_flag) + '_log.log',
                    filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )

conf_p = "config/config.json"
with open(conf_p, "r") as f:
    conf = json.load(f)
print(conf)

## Config
train_pkl_path = conf['train_pkl_path']
save_checkpoint_dir = conf['save_checkpoint_dir']
model_path = conf['model_path']
batch_size = conf['batch_size']
num_epoch = conf['num_epoch']
max_len = conf['max_len']
hidden_dim = conf['hidden_dim']
base_model = conf['base_model']
lr = conf['lr']
num_epoch_val = conf['num_epoch_val']
resume = conf['resume']
is_CRF = bool(conf['is_CRF'])
num_val_set = conf['num_val_set']
# Special Tags
START_TAG = "<START>"
STOP_TAG = "<STOP>"

## Data Format
print('data formatting...')
all_data = pkl_reader(train_pkl_path)
train_data, val_data = all_data[:-num_val_set], all_data[-num_val_set:]
train_dataset = BatchData(train_data)
val_dataset = BatchData(val_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('num samples for train:', len(train_data))
print('num_samples for val: ', len(val_data))

## Load Model
model = BertLM(max_len=max_len, hidden_dim=hidden_dim, is_CRF=is_CRF)
if resume:
    model_param = torch.load(model_path)
    model.load_state_dict(model_param['model'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('device: ' + str(device))

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
# optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
logging.info(conf)
logging.info(optimizer)

max_F1_wo = 0
acc_list = []
acc_wo_list = []
recall_list = []
recall_wo_list = []
F1_list = []
F1_wo_list = []
loss_list = []
epoch_list = []
create_dir_not_exist(save_checkpoint_dir + time_flag + '/')
print('Checkpoints will be saved into ' + save_checkpoint_dir + time_flag + '/')

total_steps = len(train_loader)
## Trainging
for epoch in tqdm(range(num_epoch)):
    step = 0
    for i, batch_data in enumerate(train_loader):
        step += 1
        model.zero_grad()

        feats = model(batch_data)
        masks = batch_data['text_info']['attention_mask']
        tags = batch_data['tags']

        loss = model.loss(feats, masks, tags)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('step: {} / {} |  epoch: {}|  loss: {}'.format(step, total_steps, epoch, loss.item()))
            logging.info('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))

    eval_info, _ = dev(model, val_loader, epoch, is_CRF)

    logging.info('------------------------------------')
    logging.info('epoch: ' + str(epoch) + ', \t_step: ' + str(step))
    logging.info(eval_info)
    logging.info('max_F1_wo: ' + str(max_F1_wo))

    acc_list.append(eval_info['acc'])
    acc_wo_list.append(eval_info['acc_wo'])
    recall_list.append(eval_info['recall'])
    recall_wo_list.append(eval_info['recall_wo'])
    F1_list.append(eval_info['F1'])
    F1_wo_list.append(eval_info['F1_wo'])
    loss_list.append(eval_info['loss'])
    epoch_list.append(epoch)

    if epoch % num_epoch_val == 0 and eval_info['F1_wo'] > max_F1_wo:
        max_F1_wo = eval_info['F1_wo']
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'lr': lr}
        save_name = save_checkpoint_dir + '/' + time_flag + '/' + 'checkpoint_epoch_' + \
                    str(epoch)[:5] + '_F1wo_' + \
                    str(eval_info['F1_wo'])[:5] + '_accwo_' + \
                    str(eval_info['acc_wo'])[:5] + '_recallwo_' + \
                    str(eval_info['recall_wo'])[:5]
        torch.save(state, save_name)
        logging.info('save model to: ' + save_name)

## Write pkl file
training_data = {'acc_list': acc_list,
                 'acc_wo_list': acc_wo_list,
                 'recall_list': recall_list,
                 'recall_wo_list': recall_wo_list,
                 'F1_list': F1_list,
                 'F1_wo_list': F1_wo_list,
                 'loss_list': loss_list,
                 'epoch_list': epoch_list,
                 'time_flag': time_flag}
print('Save training data...')
training_data_path = './training_data/train_data' + time_flag + '.pkl'
with open(training_data_path, 'wb') as f:
    pkl.dump(training_data, f)
print('Training data is saved to ' + train_pkl_path)
