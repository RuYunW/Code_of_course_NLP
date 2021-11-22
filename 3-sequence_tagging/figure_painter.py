import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import json

log_file_path = 'log/train_2021-11-18 22%3A20%3A06_log.log'

with open(log_file_path, 'r') as f:
    log_lines = f.readlines()

eval_list = []
acc_list = []
recall_list = []
F1_list = []
acc_wo_list = []
recall_wo_list = []
F1_wo_list = []
print('Data extracting...')
# for i, line in tqdm(enumerate(log_lines)):
#     if '------------------------------------' in line:
for i in tqdm(range(len(log_lines))):
    if '---------------------------' in log_lines[i]:
        text = log_lines[i+1]
        # containt = re.findall(r'INFO: {(.*?)}', text)[0]
        # print(containt)
        # exit()
        str_dict = str(text.split('INFO: ')[1])
        a = eval(str_dict)
        # print(a)
        # print(str_dict)
        # print(type(str_dict))
        # exit()
        # print(str_dict)
        # print(type(str_dict))
        # eval = eval(str_dict)
        # print(eval)

        acc_list.append(a['acc'])
        recall_list.append(a['recall'])
        F1_list.append(a['F1'])
        acc_wo_list.append(a['acc_wo'])
        recall_wo_list.append(a['recall_wo'])
        F1_wo_list.append(a['F1_wo'])
#
plt.plot(acc_list)
plt.plot(recall_list)
plt.plot(F1_list)
plt.plot(acc_wo_list)
plt.plot(recall_wo_list)
plt.plot(F1_wo_list)
plt.legend(['Acc', 'Recall', 'F1', 'Acc_wo', 'Recall_wo', 'F1_wo'])
plt.ylabel('Scores')
plt.xlabel('Epoch')
plt.show()




