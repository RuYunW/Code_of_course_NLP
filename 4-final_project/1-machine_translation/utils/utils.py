from tqdm import tqdm
import torch
import jieba
import os
import math
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

# def cal_bleu(pred_ids, label_ids, target_mask):
#     bleu_list = []
#     batch_size = label_ids.size(0)
#
#     pred_ids = pred_ids.tolist()
#     label_ids = label_ids.tolist()
#
#     pred_ids = list(map(str, pred_ids))
#     label_ids = list(map(str, label_ids))
#
#     for item in range(batch_size):
#         stop_idx = int(target_mask[item].sum())
#         bleu = sentence_bleu([label_ids[item][1: stop_idx]], pred_ids[item][1: stop_idx])
#         bleu_list.append(bleu)
#     bleu_list = np.array(bleu_list)
#     return bleu_list.mean()

def cal_acc(pred_ids, label_ids, mask):
    pred_ids = pred_ids.view(label_ids.shape).cpu()
    label_ids = label_ids.cpu()
    mask = mask.cpu()

    correct_mat = np.logical_and(pred_ids == label_ids, mask != 0)
    num_correct = correct_mat.sum()
    num_samples = (mask != 0).sum()
    acc = float(num_correct / num_samples)
    return acc

def get_results(tgt_ids, id2token, mask):
    tgt_ids = tgt_ids.view(mask.shape)  # B, L
    seq_list = []
    for i in range(int(tgt_ids.size(0))):
        stop_idx = int(mask[i].sum())
        seq = [id2token[str(int(id))] for id in tgt_ids[i][: stop_idx]]
        seq_list.append(seq)
    return seq_list

    #
    # for i, item in enumerate(tgt_ids):
    #     stop_idx = int(mask[item].sum())
    #     seq = [id2token[id] for id in item[: stop_idx]]
    #     seq_list.append(seq)
    # return seq_list



def _get_pad_mask(ids, pad_idx=2):
    mask = (ids != pad_idx).unsqueeze(-2)
    return mask

def _get_subsequent_mask(ids):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = ids.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device), diagonal=1)).bool()
    return subsequent_mask

# def val(model, data_loader, target_id2token=None, beam_search=True, beam_size=5, return_results=False, alpha=0.7):
#     model.eval()
#     bleu_scores = []
#     acc_scores = []
#     if return_results:
#         assert target_id2token is not None, 'please indicate param target_id2token'
#     result_list = []  # using if param return_result == True
#     if beam_search:
#         for batch_data in data_loader:
#             gold_ids = batch_data['target_ids']
#
#             target_mask = batch_data['target_mask_tens']
#             tgt_tens = model(batch_data)
#         pass
#
#     else:
#         for batch_data in data_loader:
#             gold_ids = batch_data['target_ids']
#             # target_mask = _get_pad_mask(???)
#             target_mask = batch_data['target_mask_tens']
#             pred = model(batch_data['source_ids'].transpose(0, 1), batch_data['target_ids'].transpose(0, 1))
#             # print(pred.shape)  # torch.Size([1024, 2004])
#             # tgt_tens = model(batch_data)
#             # tgt_tens = F.softmax(tgt_tens, dim=-1)
#             tgt_tens = F.softmax(pred, dim=-1)
#
#
#             _, tgt_ids = tgt_tens.topk(1, dim=-1, largest=True)
#
#             tgt_ids = tgt_ids.squeeze()
#
#             # BLEU
#             bleu = cal_bleu(tgt_ids, gold_ids, target_mask)
#             bleu_scores.append(bleu)
#             # ACC
#             acc = cal_acc(tgt_ids, gold_ids, target_mask)
#             acc_scores.append(acc)
#             if return_results:
#                 result_list += get_results(tgt_ids, target_id2token, target_mask)
#
#     bleu_score = np.array(bleu_scores).mean()
#     acc_score = np.array(acc_scores).mean()
#     scores = {'bleu': bleu_score, 'acc': acc_score}
#     if return_results:
#         return scores, result_list
#     return scores

# def save_prediction():
#     pass

def cal_loss(pred, gold, pad_idx=2, is_smoothing=False):
    gold = gold.contiguous().view(-1)
    if is_smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=pad_idx, reduction='sum')
    return loss


def val_acc(model, batch_data, pad_idx=2):
    model.eval()
    acc_list = []
    # for batch_data in tqdm(val_loader):
    src_ids = batch_data['source_ids']
    tgt_ids = batch_data['target_ids']
    # PE = batch_data['PE']
    pred = model(src_ids, tgt_ids)
    n_correct, n_word = cal_correct(pred, tgt_ids, pad_idx=pad_idx)
    acc = (n_correct+1) / (n_word+1)  # avoid being divided by zero
    return acc
    # acc_list.append(acc)
    # acc_list = np.array(acc_list)
    # return acc_list.mean()


def cal_correct(pred, gold, pad_idx=2):
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word


def positional_embedding(max_sen_len=64, d_model=512):
    PE = torch.zeros([max_sen_len, d_model]).to(device)
    for pos in range(max_sen_len):
        for i in range(int(d_model / 2)):
            PE[pos][2*i]   = math.sin(pos / (10000 ** ((2*i) / d_model)))
            PE[pos][2*i+1] = math.cos(pos / (10000 ** ((2*i) / d_model)))
    return PE

def from_ids_to_seq(ids, id2token):
    seq = []
    for id in ids:
        seq.append(id2token[str(id)])
    return seq


# def cal_bleu(all_pred_seq, all_label_seq):
#     bleu_list = []
#     batch_size = label_ids.size(0)
#
#     pred_ids = pred_ids.tolist()
#     label_ids = label_ids.tolist()
#
#     pred_ids = list(map(str, pred_ids))
#     label_ids = list(map(str, label_ids))
#
#     for item in range(batch_size):
#         stop_idx = int(target_mask[item].sum())
#         bleu = sentence_bleu([label_ids[item][1: stop_idx]], pred_ids[item][1: stop_idx])
#         bleu_list.append(bleu)
#     bleu_list = np.array(bleu_list)
#     return bleu_list.mean()

def not_pad(token):
    return token != '<PAD>'

def cal_bleu(pred_seq: list, label_seq):
    pred_seq = list(filter(not_pad, pred_seq))
    label_seq = list(filter(not_pad, label_seq))
    bleu = sentence_bleu([label_seq], pred_seq)
    return bleu

def write_results(acc, bleu, sources, labels, results, save_results_path):
    print('Writing result file...')
    with open(save_results_path, 'w', encoding='utf-8') as fw:
        fw.write('acc: {}, bleu: {}'.format(acc, bleu))
        fw.write('\n\n')
        for i in tqdm(range(len(results))):
            source_seq = ' '.join(list(filter(not_pad, sources[i])))
            label_seq = ''.join(list(filter(not_pad, labels[i])))
            pred_seq = ''.join(list(filter(not_pad, results[i])))

            fw.writelines(source_seq)
            fw.write('\n')
            fw.writelines(label_seq)
            fw.write('\n')
            fw.writelines(pred_seq)
            fw.write('\n\n')
    print('Result file writing completed.')
    print('File is saved to: ' + save_results_path)





