import time
import json
import torch
import logging
import pickle as pkl
from tqdm import tqdm
from torch.optim import Adam
from dataset import BatchData
from torch.utils.data import DataLoader
from model.transformer import Transformer
from model.transformer import ScheduledOptim
from utils.utils import val_acc, cal_loss, create_dir_not_exist, save_np_file


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
results_dir = conf['results_dir']
# hyper-parameters
batch_size = conf['batch_size']
num_epoch = conf['num_epoch']
# lr = conf['lr']
max_len = conf['max_len']
en_vocab_size = conf['en_vocab_size']
zh_vocab_size = conf['zh_vocab_size']
num_encoder_layers = conf['num_encoder_layers']
num_decoder_layers = conf['num_decoder_layers']
d_model = conf['d_model']
num_heads = conf['num_heads']
d_inner = conf['d_inner']

num_vals = 1000

is_resume = bool(conf['is_resume'])
num_print = conf['num_print']
save_step = conf['save_step']

if is_resume:
    model_path = conf['model_path']


## Log
logging.basicConfig(level=logging.DEBUG,
                    filename=log_dir+'train_' + str(time_flag) + '.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
# read token2id
with open(vocab_dir+'token2id_en.json', 'r') as f:
    token2id_en = json.load(f)
with open(vocab_dir+'token2id_zh.json', 'r') as f:
    token2id_zh = json.load(f)
with open(vocab_dir+'id2token_zh.json', 'r') as f:
    id2token_zh = json.load(f)
# read data
with open(ids_dir+'train_en_ids.pkl', 'rb') as f:
    train_en_ids = pkl.load(f)
with open(ids_dir+'train_zh_ids.pkl', 'rb') as f:
    train_zh_ids = pkl.load(f)
assert len(train_en_ids) == len(train_zh_ids), 'Train: The length of en and zh are not equal. '
with open(ids_dir+'val_en_ids.pkl', 'rb') as f:
    val_en_ids = pkl.load(f)[:num_vals]
with open(ids_dir+'val_zh_ids.pkl', 'rb') as f:
    val_zh_ids = pkl.load(f)[:num_vals]
assert len(val_en_ids) == len(val_zh_ids), 'Val: The length of en and zh are not equal. '

train_dataset = BatchData(train_en_ids, train_zh_ids, en_vocab_size, zh_vocab_size, max_len, token2id_en, token2id_zh)
val_dataset = BatchData(val_en_ids, val_zh_ids, en_vocab_size, zh_vocab_size, max_len, token2id_en, token2id_zh)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print('num train samples: ', len(train_en_ids))
print('num val samples: ', len(val_en_ids))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Transformer(num_enc_layers=num_encoder_layers, num_dec_layers=num_decoder_layers, max_sen_len=max_len,
                    d_model=d_model, input_size=en_vocab_size+4, output_size=zh_vocab_size+4,
                    num_heads=num_heads, d_inner=d_inner)
model.to(device)
print('device: ' + str(device))

optimizer = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul=2.0, d_model=512, n_warmup_steps=4000)

create_dir_not_exist(checkpoints_dir + time_flag + '/')
print('Checkpoints will be saved into ' + checkpoints_dir + time_flag + '/')
create_dir_not_exist(results_dir + time_flag + '/')
print('Results will be saved at: ' + results_dir + time_flag + '/')



## train
model.train()
total_steps = len(train_loader)
val_best_acc = 0
val_best_loss = 10000
val_iter_data = iter(val_loader)
train_accs = []
val_accs = []
val_losss = []
val_bleus = []
train_losss = []

for epoch in tqdm(range(num_epoch)):
    step = 0
    # train
    for i, batch_data in enumerate(train_loader):
        model.zero_grad()
        # inputs
        src_ids = batch_data['source_ids']
        tgt_ids = batch_data['target_ids']
        # predict
        pred = model(src_ids, tgt_ids)
        # scores
        loss = cal_loss(pred, tgt_ids, is_smoothing=True)
        acc = val_acc(model, batch_data)
        train_losss.append(loss)
        train_accs.append(acc)
        # print
        if step % num_print == 0:
            print_info = 'Training: epoch: {} / {}|    step: {} / {}|    loss: {}|    Acc: {}%'.format(
                epoch, num_epoch, step, total_steps, float('%.2f'%loss.item()), float('%.2f'%(acc*100)))
            print(print_info)
            logging.info(print_info)
        # backward
        loss.backward()
        optimizer.step_and_update_lr()

        # val
        if step % save_step == 0:
            model.eval()
            # inputs
            val_batch_data = val_iter_data.next()
            val_src_ids, val_tgt_ids = val_batch_data['source_ids'], val_batch_data['target_ids']
            # predict
            val_pred = model(val_src_ids, val_tgt_ids)
            # scores
            val_set_loss = cal_loss(val_pred, val_tgt_ids, is_smoothing=True)
            val_set_acc, val_set_bleu = val_acc(model, val_batch_data, cal_bleu=True)
            print('Val: epoch: {} / {}|    step: {} / {}|    Loss: {}|    Acc: {}%|    BLEU: {}%|'.format(
                    epoch, num_epoch, step, total_steps, float('%.2f'%val_set_loss.item()), float('%.2f'%(val_set_acc*100)), float('%.2f'%(val_set_bleu*100))))
            val_losss.append(val_set_loss)
            val_accs.append(val_set_acc)
            val_bleus.append(val_set_bleu)
            # save train acc
            save_scores_dir = results_dir + time_flag
            save_np_file(save_scores_dir+'/train_acc_scores.npy', train_accs)
            save_np_file(save_scores_dir+'/train_loss.npy', train_losss)
            save_np_file(save_scores_dir+'/val_acc_scores.npy', val_accs)
            save_np_file(save_scores_dir+'/val_loss.npy', val_losss)
            save_np_file(save_scores_dir+'/val_bleu_scores.npy', val_bleus)
            print('Acc and loss of training and val has been saved at: ' + save_scores_dir + '/')

            # if best, save
            if val_best_acc <= val_set_acc or val_best_loss >= val_set_loss.item():
                # update
                val_best_acc = max(val_best_acc, val_set_acc)
                val_best_loss = min(val_best_loss, val_set_loss.item())
                # logging
                logging.info('--------------------------------')
                logging.info('Val: epoch: {} / {}|    step: {} / {}|    acc: {}|    loss: {}|'.format(
                    epoch, num_epoch, step, total_steps, val_set_acc, val_set_loss.item()))
                # save
                # lr = optimizer._optimizer.param_groups[0]['lr']
                state = {'model': model.state_dict(), 'optimizer': optimizer, 'time_flag': str(time_flag)}
                save_name = checkpoints_dir + time_flag + '/' + 'epoch_'+str(epoch)+'_step_'+str(step)\
                            +'_acc_' + str(val_set_acc)[:5] + '_loss_' + str(val_set_loss.item())[:7] + '.checkpoint'
                torch.save(state, save_name)
                print('checkpoint has been saved at: ' + save_name)

            model.train()
        step += 1

# print final info
print('best acc: {}|    best loss: {}'.format(val_best_acc, val_best_loss))
print('time_flag: ' + str(time_flag))

