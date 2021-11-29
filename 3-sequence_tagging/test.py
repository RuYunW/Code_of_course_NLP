import torch
import json
# import logging
import time
from utils import pkl_reader, dev, save_results
from dataset import BatchData
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.bert_lm import BertLM

torch.manual_seed(1)
# time_flag = time.strftime("%Y-%m-%d %H:%M:%S")

conf_p = "config/config.json"
with open(conf_p, "r") as f:
    conf = json.load(f)
print(conf)

## Config
test_pkl_path = conf['test_pkl_path']
# save_checkpoint_dir = conf['save_checkpoint_dir']
model_path = conf['model_path']
results_dir = conf['results_dir']
time_flag = str(model_path.split('/')[-2])
print('time flag: ' + time_flag)
batch_size = conf['batch_size']
is_CRF = conf['is_CRF']
# num_epoch = conf['num_epoch']
lr = conf['lr']
result_save_path = results_dir + 'results_' + str(time_flag) + '.txt'
test_text_path = conf['test_text_path']
# num_epoch_val = conf['num_epoch_val']
# resume = conf['resume']

## Data Format
test_data = pkl_reader(test_pkl_path)
test_dataset = BatchData(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Load Model
model = BertLM(is_CRF=is_CRF)
model_param = torch.load(model_path)
model.load_state_dict(model_param['model'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print('device: ' + str(device))



eval_info, labels = dev(model=model, dev_loader=test_loader, is_CRF=is_CRF)

acc, recall, F1 = eval_info['acc'], eval_info['recall'], eval_info['F1']
acc_wo, recall_wo, F1_wo = eval_info['acc_wo'], eval_info['recall_wo'], eval_info['F1_wo']
print('eval  \n|  acc: {}\n|  recall: {}\n|  F1: {}\n|  acc_w/o_O: {}\n|  recall_w/o_O: {}\n|  F1_w/o_O: {}'.format(
            acc, recall, F1, acc_wo, recall_wo, F1_wo))
save_results(eval_info, labels, test_text_path, result_save_path)





