import matplotlib.pyplot as plt
import json
import pickle as pkl

conf_p = "config/config.json"
with open(conf_p, "r") as fc:
    conf = json.load(fc)

## Config
model_path = conf['model_path']
save_figure_dir = conf['save_figure_dir']
time_flag = str(model_path.split('/')[-2])
training_data_path = './training_data/train_data' + time_flag + '.pkl'
with open(training_data_path, 'rb') as ft:
    training_data = pkl.load(ft)

## Data
acc_list = training_data['acc_list']
acc_wo_list = training_data['acc_wo_list']
recall_list = training_data['recall_list']
recall_wo_list = training_data['recall_wo_list']
F1_list = training_data['F1_list']
F1_wo_list = training_data['F1_wo_list']
loss_list = training_data['loss_list']
epoch_list = training_data['epoch_list']
time_flag = training_data['time_flag']

## Painting figures
# Scores
plt.plot(acc_list)
plt.plot(recall_list)
plt.plot(F1_list)
plt.plot(acc_wo_list)
plt.plot(recall_wo_list)
plt.plot(F1_wo_list)
plt.title('Scores in training')
plt.legend(['Acc+O', 'Recall+O', 'F1+O', 'Acc-O', 'Recall-O', 'F1-O'])
plt.ylabel('Scores')
plt.xlabel('Epoch')
plt.savefig(save_figure_dir+time_flag+'_training_data.png')
plt.clf()
# loss
plt.plot(loss_list)
plt.legend(['loss'])
plt.title('Loss value in training')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.savefig(save_figure_dir+time_flag+'_loss.png')

print('Figures has been successfully saved. ')
