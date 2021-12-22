from tqdm import tqdm
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


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


def _get_pad_mask(ids, pad_idx=2):
    mask = (ids != pad_idx).unsqueeze(-2)
    return mask

def _get_subsequent_mask(ids):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = ids.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device), diagonal=1)).bool()
    return subsequent_mask


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


def val_acc(model, batch_data, pad_idx=2, cal_bleu=False):
    model.eval()
    src_ids = batch_data['source_ids']
    tgt_ids = batch_data['target_ids']
    pred = model(src_ids, tgt_ids)
    n_correct, n_word = cal_correct(pred.cpu(), tgt_ids.cpu(), pad_idx=pad_idx)
    acc = (n_correct+1) / (n_word+1)  # avoid being divided by zero
    if cal_bleu:
        bleu = cal_batch_bleu(pred.cpu(), tgt_ids.cpu())
        return acc, bleu
    return acc


def cal_batch_bleu(pred, gold, pad_idx=2):
    batch_size = int(gold.size(0))
    pred = pred.max(1)[1].view(batch_size, -1)
    pred = list(filter(not_pad, pred.tolist()))
    gold = list(filter(not_pad, gold.tolist()))
    bleu_list = [sentence_bleu([list(map(str, gold[i]))], list(map(str, pred[i]))) for i in range(batch_size)]
    return np.array(bleu_list).mean()

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

def save_np_file(file_path, scores_list):
    scores_list = np.array(scores_list)
    np.save(file_path, np.array(scores_list))
    # print('Training loss has been saved at: ' + file_path)





