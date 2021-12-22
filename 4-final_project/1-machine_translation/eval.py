import json
import pickle as pkl
from dataset import BatchData
from torch.utils.data import DataLoader
from model.transformer import Transformer
import torch
from model.translator import Translator
from tqdm import tqdm
from utils.utils import from_ids_to_seq, cal_correct, cal_bleu, write_results
import numpy as np
# from utils.utils import val

conf_p = "config/config.json"
with open(conf_p, "r") as f:
    conf = json.load(f)
print(conf)

# Config
test_en_set_path = conf['test_en_set_path']
test_zh_set_path = conf['test_zh_set_path']
model_path = conf['model_path']
max_len = conf['max_len']
batch_size = conf['batch_size']
vocab_dir = conf['vocab_dir']
ids_dir = conf['ids_dir']
results_dir = conf['results_dir']
en_vocab_size = conf['en_vocab_size']
zh_vocab_size = conf['zh_vocab_size']
beam_size = conf['beam_size']



with open(vocab_dir+'token2id_en.json', 'r') as f:
    token2id_en = json.load(f)
with open(vocab_dir+'token2id_zh.json', 'r') as f:
    token2id_zh = json.load(f)
with open(vocab_dir+'id2token_en.json', 'r') as f:
    id2token_en = json.load(f)
with open(vocab_dir+'id2token_zh.json', 'r') as f:
    id2token_zh = json.load(f)
with open(ids_dir+'test_en_ids.pkl', 'rb') as f:
    test_en_ids = pkl.load(f)
with open(ids_dir+'test_zh_ids.pkl', 'rb') as f:
    test_zh_ids = pkl.load(f)

test_en_ids = test_en_ids[:20]
test_zh_ids = test_zh_ids[:20]
test_dataset = BatchData(test_en_ids, test_zh_ids, en_vocab_size, zh_vocab_size, max_len, token2id_en, token2id_zh)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Transformer(max_sen_len=max_len, input_size=en_vocab_size+4, output_size=zh_vocab_size+4)

model_param = torch.load(model_path)
model.load_state_dict(model_param['model'])
time_flag = model_param['time_flag']
model.to(device)
print('device: ' + str(device))
model.eval()

# for batch_data in test_loader:
#     src_ids = batch_data['source_ids']
#     tgt_ids = batch_data['target_ids']
#     pred = model(src_ids, tgt_ids)
#     exit()

translator = Translator(model, id2token_zh, max_sen_len=max_len, beam_size=beam_size)
translator.to(device)

sources = []
labels = []
results = []

acc_scores = []
bleu_scores = []
for batch_data in tqdm(test_loader):
    src_ids = batch_data['source_ids']
    label_ids = batch_data['target_ids']
    pred_ids = translator.translate_sentence(batch_data)

    source_seq = from_ids_to_seq(src_ids.squeeze().tolist(), id2token_en)
    label_seq = from_ids_to_seq(label_ids.squeeze().tolist(), id2token_zh)
    pred_seq = from_ids_to_seq(pred_ids, id2token_zh)

    sources.append(source_seq)
    labels.append(label_seq)
    results.append(pred_seq)


    n_correct, n_word = cal_correct(torch.tensor(pred_ids).unsqueeze(0).to(device), label_ids)
    acc = (n_word+1) / (n_correct+1)  # smooth
    bleu = cal_bleu(pred_seq, label_seq)

    acc_scores.append(acc)
    bleu_scores.append(bleu)

acc_scores = np.array(acc_scores)
bleu_scores = np.array(bleu_scores)
acc = acc_scores.mean()
bleu = bleu_scores.mean()

save_results_path = results_dir + 'results_' + time_flag + '_acc_' + str(acc) + '_bleu_' + str(bleu) + '.txt'
write_results(acc, bleu, sources, labels, results, save_results_path)





