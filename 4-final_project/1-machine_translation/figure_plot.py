import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import re

# Config
conf_p = "config/config.json"
with open(conf_p, "r") as fc:
    conf = json.load(fc)
model_path = conf['model_path']
results_dir = conf['results_dir']
figures_dir = conf['figures_dir']

model_param = torch.load(model_path, map_location=lambda storage, loc: storage)  # to(cpu)
time_flag = model_param['time_flag']

# val_acc_save_name = results_dir + time_flag + '/val_acc_scores' + '_epoch_' + str(epoch) \
#                                 + '_step_' + str(step) + '.npy'

acc_save_name_list = os.listdir(results_dir + time_flag)
acc_save_name_list.sort(key= lambda x:int(re.findall(r'scores_epoch_0_step_(.*?).npy', x)[0]))
val_acc_scores_name, train_acc_scores_name = acc_save_name_list[-2], acc_save_name_list[-1]

val_accs = np.load(results_dir + time_flag+'/'+val_acc_scores_name)
train_accs = np.load(results_dir + time_flag + '/' + train_acc_scores_name)


# print(val_accs)
# print(len(val_accs))
# x = np.arange(0, len(val_accs)*200, 200)
# print(x)

plt.plot(train_accs)
plt.plot(np.arange(0, len(val_accs)*200, 200), val_accs, color='red', linewidth='1')
plt.legend(['train', 'val'])
plt.xlabel('Steps')
plt.ylabel('Acc')
plt.title('Acc scores in training.')
# plt.xticks()
plt.savefig(figures_dir+'/'+val_acc_scores_name+'.png')
# print(acc_save_name_list)
# val_accs = np.load()

