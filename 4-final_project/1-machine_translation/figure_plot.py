import matplotlib.pyplot as plt
import numpy as np
import json
# import torch
# import os
# import re

# Config
conf_p = "config/config.json"
with open(conf_p, "r") as fc:
    conf = json.load(fc)
# model_path = conf['model_path']
results_dir = conf['results_dir']
figures_dir = conf['figures_dir']

# time_flag = '2021-12-24 12%3A20%3A05'
# time_flag = '2021-12-24 12%3A38%3A18'
# time_flag = '2021-12-24 21%3A44%3A19'
# time_flag = '2021-12-24 22%3A26%3A57'
time_flag = '2021-12-29 12%3A08%3A44'

results_time_flag = './results/'+time_flag + '/'
train_acc_path = results_time_flag + 'train_acc_scores.npy'
train_acc = np.load(train_acc_path)

# train_loss_path = time_flag + 'train_loss.npy'
# train_loss = np.load(train_loss_path, allow_pickle=True)

val_acc_path = results_time_flag + 'val_acc_scores.npy'
val_acc = np.load(val_acc_path)

val_bleu_path = results_time_flag + 'val_bleu_scores.npy'
val_bleu = np.load(val_bleu_path)

# print(max(val_bleu))
# print(val_bleu.mean())
# val_loss_path = time_flag + 'val_loss.npy'
# val_loss = np.load(val_loss_path)


#
# train_acc = read_npy(train_acc_path)
# train_loss = read_npy(train_loss_path)
# val_acc = read_npy(val_acc_path)
# val_bleu = read_npy(val_bleu_path)
# val_loss = read_npy(val_loss_path)
# exit()
# model_param = torch.load(model_path, map_location=lambda storage, loc: storage)  # to(cpu)
# time_flag = model_param['time_flag']

# print(len(train_acc))
train_x = np.arange(0, len(train_acc))
val_x = np.arange(0, len(val_acc)*3500, 3500)


plt.plot(train_x, train_acc)
# plt.plot(train_x, train_loss)
plt.plot(val_x, val_acc)
plt.plot(val_x, val_bleu)
# plt.plot(val_x, val_loss)
# plt.plot(, val_accs, color='red', linewidth='1')
plt.legend(['train acc', 'val acc', 'val bleu'])
plt.xlabel('Steps')
plt.ylabel('Score')
plt.title('Acc and BLEU scores in training.')
# plt.xticks()
# from utils.utils import create_dir_not_exist
# create_dir_not_exist()
plt.savefig(figures_dir + 'results_plot_' + time_flag + '.png')
# print(acc_save_name_list)
# val_accs = np.load()
