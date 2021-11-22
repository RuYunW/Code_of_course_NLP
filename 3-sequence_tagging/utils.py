import pickle as pkl
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

def pkl_reader(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_lines = pkl.load(f)
    return pkl_lines


def evaluation(pred, true, neglect=False):
    assert len(pred) == len(true)
    tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
    true_labels = np.array(true)
    pred_labels = np.array(pred)
    if neglect:
        num_correct = (np.logical_and(true_labels == pred_labels,
                                      np.logical_and(true_labels != tag_to_ix['O'], true_labels != tag_to_ix['PAD']))
                       ).astype(np.int).sum()
        num_pred = (np.logical_and(pred_labels != tag_to_ix['O'], pred_labels != tag_to_ix['PAD'])).astype(np.int).sum()
        num_true = (np.logical_and(true_labels != tag_to_ix['O'], true_labels != tag_to_ix['PAD'])).astype(np.int).sum()
        acc = (num_correct+1) / (num_pred+1)
        recall = (num_correct+1) / (num_true+1)

    else:
        num_correct = (np.logical_and(true_labels == pred_labels, true_labels != tag_to_ix['PAD'])).astype(np.int).sum()
        num_pred = (pred_labels != tag_to_ix['PAD']).astype(np.int).sum()
        num_true = (true_labels != tag_to_ix['PAD']).astype(np.int).sum()
        acc = (num_correct+1) / (num_pred+1)
        recall = (num_correct+1) / (num_true+1)
    F1 = 2 * acc * recall / (acc + recall)
    return acc, recall, F1


def dev(model, dev_loader, epoch=0):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    sentence_len = []
    length = 0

    for i, batch_data in enumerate(dev_loader):
        feats = model(batch_data)
        masks = batch_data['text_info']['attention_mask']
        tags = batch_data['tags']
        length += tags.size(0)

        path_score, best_path = model.crf(feats, masks.bool())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path.cpu().tolist()])
        true.extend([t for t in tags.cpu().tolist()])

    acc, recall, F1 = evaluation(pred, true)
    acc_wo, recall_wo, F1_wo = evaluation(pred, true, neglect=True)
    print(
        'eval  epoch: {}|  loss: {}|  acc: {}|  recall: {}|  F1: {}|  acc_w/o_O: {}|  recall_w/o_O: {}|  F1_w/o_O: {}'.format(
            epoch, eval_loss / length, acc, recall, F1, acc_wo, recall_wo, F1_wo))

    model.train()
    return {'loss': eval_loss, 'acc': acc, 'recall': recall, 'F1': F1, 'acc_wo': acc_wo,
            'recall_wo': recall_wo, 'F1_wo': F1_wo}, {'pred': pred, 'true': true, }


def save_results(eval_info, labels, text_path, save_path):
    tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
    ix_to_tag = {}
    for t in tag_to_ix.keys():
        ix = tag_to_ix[t]
        ix_to_tag[ix] = str(t)
    pred = labels['pred']
    true = labels['true']
    all_pred_tag = []
    all_true_tag = []
    all_text_t = []

    ## Writing File
    with open(text_path, 'r', encoding='utf-8') as ft:
        text_lines = ft.readlines()

    print(len(text_lines))
    print(len(pred))
    print(len(true))
    assert len(text_lines) == len(pred), 'num sample not match'

    for i, text in tqdm(enumerate(text_lines)):
        all_pred_tag.append(' '.join([ix_to_tag[id] for id in pred[i]]))
        all_true_tag.append(' '.join([ix_to_tag[id] for id in true[i]]))
        all_text_t.append(text[:-1])

    print('Writing result file...')
    with open(save_path, 'w', encoding='utf-8') as fw:
        fw.write('acc: {}, recall: {}, F1: {}'.format(eval_info['acc'], eval_info['recall'], eval_info['F1']))
        fw.writ('\n')
        fw.write('acc_wo: {}, recall_wo: {}, F1_wo: {}'.format(eval_info['acc_wo'], eval_info['recall_wo'], eval_info['F1_wo']))
        fw.writ('\n')
        for i in range(len(text_lines)):
            fw.writelines(all_text_t[i])
            fw.writelines('\n')
            fw.writelines(all_pred_tag[i])
            fw.writelines('\n')
            fw.writelines(all_true_tag[i])
            fw.write('\n')

    print('Result file writing completed.')
    print('File is saved to: ' + save_path)



