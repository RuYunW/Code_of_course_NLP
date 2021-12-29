import re
import time
import json
import math
import torch
import logging
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torch.optim import Adam
from dataset import BatchData
from torch.utils.data import DataLoader
from model.translator import Translator
from model.transformer import Transformer
# from model.zhihu import Transformer
# from transformer.Models import Transformer
from model.transformer import ScheduledOptim
from utils.utils import create_dir_not_exist, save_np_file, get_iter, cal_performance, cal_batch_bleu, val


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
warmup_step = conf['warmup_step']
# lr = conf['lr']
max_len = conf['max_len']
en_vocab_size = conf['en_vocab_size']
zh_vocab_size = conf['zh_vocab_size']
num_encoder_layers = conf['num_encoder_layers']
num_decoder_layers = conf['num_decoder_layers']
d_model = conf['d_model']
num_heads = conf['num_heads']
d_inner = conf['d_inner']
beam_size = conf['beam_size']

num_vals = 2000

is_resume = bool(conf['is_resume'])
num_print = conf['num_print']
save_step = conf['save_step']

## Log
logging.basicConfig(level=logging.DEBUG,
                    filename=log_dir+'train_' + str(time_flag) + '.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
logging.info(conf)
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

# train_dataset = BatchData(train_en_ids, train_zh_ids, max_len)
val_dataset = BatchData(val_en_ids, val_zh_ids, max_len)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print('num train samples: ', len(train_en_ids))
print('num val samples: ', len(val_en_ids))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
model = Transformer(num_enc_layers=num_encoder_layers, num_dec_layers=num_decoder_layers, max_sen_len=max_len,
                    d_model=d_model, input_size=en_vocab_size+4, output_size=zh_vocab_size+4,
                    num_heads=num_heads, d_inner=d_inner)
# model = Transformer(src_pad_idx=2, trg_pad_idx=2, trg_sos_idx=0, enc_voc_size=en_vocab_size+4, dec_voc_size=zh_vocab_size+4,
#                     d_model=d_model, n_head=num_heads, max_len=max_len,
#                  ffn_hidden=2048, n_layers=6, drop_prob=0.1, device=device)
# model = Transformer(n_src_vocab=en_vocab_size+4, n_trg_vocab=zh_vocab_size+4,
#                     src_pad_idx=2, trg_pad_idx=2)
optimizer = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul=2.0, d_model=512, n_warmup_steps=warmup_step)

translator = Translator(model, id2token_zh, max_sen_len=max_len, beam_size=beam_size)
translator.to(device)

# total_steps = len(train_loader)
total_steps = math.ceil(len(train_en_ids) / batch_size)
# val_total_steps = len(val_loader)
val_total_steps = math.ceil(len(val_en_ids) / batch_size)
val_best_acc = 0
val_best_loss = 10000
val_best_bleu = 0
val_iter_data = iter(val_loader)
train_accs = []
val_accs = []
val_losss = []
val_bleus = []
train_losss = []


start_step = 0
start_epoch = 0
resume_counter = 0
val_step = 0


if is_resume:
    model_path = conf['model_path']
    model_param = torch.load(model_path)
    model.load_state_dict(model_param['model'])

    time_flag = model_param['time_flag']
    # start_step = int(re.findall(r'_step_(.*?)_', model_path)[0])
    start_epoch = int(re.findall(r'epoch_(.*?)_', model_path)[0])
    print('Resume training from model: '+model_path)
    print('Start epoch: {}'.format(start_epoch))

if is_resume:
    train_losss = np.load(results_dir+time_flag+'/train_loss.npy').tolist()
    train_accs = np.load(results_dir+time_flag+'/train_acc_scores.npy').tolist()
    val_losss = np.load(results_dir+time_flag+'/val_loss.npy').tolist()
    val_accs = np.load(results_dir+time_flag+'/val_acc_scores.npy').tolist()
    val_bleus = np.load(results_dir+time_flag+'/val_bleu_scores.npy').tolist()

model.to(device)
print('device: ' + str(device))

create_dir_not_exist(checkpoints_dir + time_flag + '/')
print('Checkpoints will be saved into ' + checkpoints_dir + time_flag + '/')
create_dir_not_exist(results_dir + time_flag + '/')
print('Results will be saved at: ' + results_dir + time_flag + '/')

model.train()
for epoch in tqdm(range(start_epoch, num_epoch)):
    step = 0
    # train
    for i in range(total_steps):
        batch_data = get_iter(train_en_ids[i*batch_size: (i+1)*batch_size],
                              train_zh_ids[i*batch_size: (i+1)*batch_size],
                              max_sen_len=max_len, batch_size=batch_size, shuffle=True)
        model.zero_grad()
        # inputs
        src_ids = batch_data['source_ids']
        tgt_ids = batch_data['target_ids']
        gold = tgt_ids[:, 1:].contiguous().view(-1)
        # predict
        pred = model(src_ids, tgt_ids[:, :-1])
        # scores
        loss, n_correct, n_word = cal_performance(
            pred, gold, 2, smoothing=True)
        acc = n_correct / (n_word+0.001)
        train_losss.append(loss.item())
        train_accs.append(acc)
        # print
        if step % num_print == 0:
            print_info = 'Training: epoch: {} / {}|    step: {} / {}|    loss: {}|    Acc: {}%'.format(
                epoch, num_epoch, step, total_steps, float('%.2f' % loss.item()), float('%.2f' % (acc*100)))
            print(print_info)
            logging.info(print_info)
        # backward
        loss.backward()
        optimizer.step_and_update_lr()

        # val
        if step != 0 and step % save_step == 0:
            val_info = val(model, val_en_ids, val_zh_ids, batch_size, max_len)
            logging.info(val_info['val_logging_info'])
            print(val_info['val_logging_info'])
            #
            # val_step = val_step  % val_total_steps
            # val_batch_data = get_iter(val_en_ids[val_step*batch_size: (val_step+1)*batch_size],
            #                           val_zh_ids[val_step*batch_size: (val_step+1)*batch_size],
            #                           max_sen_len=max_len, batch_size=batch_size, shuffle=False)
            # val_src_ids, val_tgt_ids = val_batch_data['source_ids'], val_batch_data['target_ids']
            # val_gold = val_tgt_ids[:, 1:].contiguous().view(-1)
            # model.eval()
            # with torch.no_grad():
            #     val_pred = model(val_src_ids, val_tgt_ids[:, :-1])
            # val_step += 1
            # # scores
            # val_set_loss, val_n_correct, val_n_word = cal_performance(val_pred, val_gold, 2, smoothing=True)
            # val_set_acc = val_n_correct / (val_n_word + 0.001)
            # val_set_bleu = cal_batch_bleu(val_pred.cpu(), val_gold.cpu(), batch_size=batch_size)
            # val_print_info = 'Val: epoch: {} / {}|    step: {} / {}|    Loss: {}|    Acc: {}%|    BLEU: {}%|'.format(
            #         epoch, num_epoch, step, total_steps,
            #     float('%.2f'%val_set_loss.item()), float('%.2f'%(val_set_acc*100)), float('%.2f'%(val_set_bleu*100)))
            # print(val_print_info)
            # logging.info(val_print_info)
            val_losss.append(val_info['val_mean_loss'])
            val_accs.append(val_info['val_mean_acc'])
            val_bleus.append(val_info['val_mean_bleu'])
            # save train acc
            save_scores_dir = results_dir + time_flag
            save_np_file(save_scores_dir+'/train_acc_scores.npy', train_accs)
            save_np_file(save_scores_dir+'/train_loss.npy', train_losss)
            save_np_file(save_scores_dir+'/val_acc_scores.npy', val_accs)
            save_np_file(save_scores_dir+'/val_loss.npy', val_losss)
            save_np_file(save_scores_dir+'/val_bleu_scores.npy', val_bleus)
            print('Acc and loss of training and val has been saved at: ' + save_scores_dir + '/')

            # if best, save
            if val_best_acc < val_info['val_mean_acc'] or val_best_loss > val_info['val_mean_loss'] or val_best_bleu < val_info['val_mean_bleu'] \
                    or epoch == (num_epoch-1):
                # update
                val_best_acc = max(val_best_acc, val_info['val_mean_acc'])
                val_best_loss = min(val_best_loss, val_info['val_mean_loss'])
                val_best_bleu = max(val_best_bleu, val_info['val_mean_bleu'])
                # # logging
                # logging.info('--------------------------------')
                # logging.info('Val: epoch: {} / {}|    step: {} / {}|    acc: {}|    bleu: {}|    loss: {}|'.format(
                #     epoch, num_epoch, step, total_steps, val_set_acc, val_set_bleu, val_set_loss.item()))
                # save
                # lr = optimizer._optimizer.param_groups[0]['lr']
                state = {'model': model.state_dict(), 'optimizer': optimizer, 'time_flag': str(time_flag)}
                checkpoint_name = checkpoints_dir + time_flag + '/' + 'epoch_' + str(epoch) + '_step_' + str(step) \
                                  +'_acc_' + str(val_info['val_mean_acc'])[:5] + '_bleu_' + str(val_info['val_mean_bleu'])[:5] \
                                  +'_loss_' + str(val_info['val_mean_loss'])[:7] + '.checkpoint'
                torch.save(state, checkpoint_name)
                print('checkpoint has been saved at: ' + checkpoint_name)
            model.train()
        step += 1

# print final info
print('best acc: {}|    best bleu: {}|    best loss: {}'.format(val_best_acc, val_best_bleu, val_best_loss))
print('time_flag: ' + str(time_flag))

