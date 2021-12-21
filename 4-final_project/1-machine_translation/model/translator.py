import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nltk.translate.bleu_score import sentence_bleu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def positional_embedding(max_sen_len=64, d_model=512):
    PE = torch.zeros([max_sen_len, d_model]).to(device)
    for pos in range(max_sen_len):
        for i in range(int(d_model / 2)):
            PE[pos][2*i]   = math.sin(pos / (10000 ** ((2*i) / d_model)))
            PE[pos][2*i+1] = math.cos(pos / (10000 ** ((2*i) / d_model)))
    return PE

class Translator(nn.Module):
    def __init__(self, model, id2token, max_sen_len=64, beam_size=5, sos_idx=0, eos_idx=1, pad_idx=2, alpha=0.7):
        super(Translator, self).__init__()
        self.alpha = alpha
        self.max_sen_len = max_sen_len
        self.encoder_PE = positional_embedding()
        self.id2token = id2token
        self.beam_size = beam_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[sos_idx]]))
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_sen_len), pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.sos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_sen_len + 1, dtype=torch.long).unsqueeze(0))

    def _get_pad_mask(self, ids):
        mask = (ids != self.pad_idx).unsqueeze(-2)
        return mask

    def _get_subsequent_mask(self, ids):
        ''' For masking out the subsequent info. '''
        sz_b, len_s = ids.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=ids.device), diagonal=1)).bool()
        return subsequent_mask

    def _model_decode(self, tgt_ids, hidden_mat, src_mask, PE):
        # Decoder Embedding
        tgt_emb = self.model.tgt_embedding(tgt_ids)
        # print(tgt_ids.shape)
        tgt_pos = PE
        # print(PE.shape)
        # exit()

        # print(tgt_emb.shape)
        # print(PE.shape)
        # exit()
        tgt_feats = tgt_emb + tgt_pos[:, :tgt_emb.size(1), :]
        # print(tgt_emb.shape)
        # print(tgt_pos.shape)
        # exit()
        # tgt_feats = tgt_emb + tgt_pos

        tgt_feats = self.model.dropout(tgt_feats)
        tgt_feats = self.model.layer_norm(tgt_feats)
        # print(tgt_feats.shape)

        # Mask
        tgt_mask = self._get_subsequent_mask(tgt_ids)
        # print(tgt_mask.shape)
        # Decoder
        for decoder_block in self.model.decoder:
            # print(tgt_feats.shape)  # torch.Size([1, 1, 512])
            # print(tgt_mask.shape)  # torch.Size([1, 1, 1])
            # print(src_mask.shape)  # torch.Size([1, 1, 64])
            # exit()
            tgt_feats = decoder_block(tgt_feats, hidden_mat, slf_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask)
        tgt_feats = self.model.linear(tgt_feats)
        return F.softmax(tgt_feats, dim=-1)

    def _get_init_state(self, src_ids, src_mask, PE):
        # Embedding
        src_emb = self.model.src_embedding(src_ids)
        src_pos = self.encoder_PE
        src_feats = src_emb + src_pos
        src_feats = self.model.dropout(src_feats)
        src_feats = self.model.layer_norm(src_feats)
        # Encoder
        for encoder_block in self.model.encoder:
            enc_output, hidden_mat = encoder_block(src_feats, src_mask)
        # print(self.init_seq.shape)  # [1, 1]
        # exit()
        # PE = positional_embedding(self.init_seq.shape[1])
        # print(PE.shape)
        # print(666)
        # exit()
        # Decode
        dec_output = self._model_decode(self.init_seq, hidden_mat, src_mask, PE=PE)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(self.beam_size)

        scores = torch.log(best_k_probs).view(self.beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(self.beam_size, 1, 1)
        return enc_output, gen_seq, scores, hidden_mat

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        # beam_size = self.beam_size
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(self.beam_size)
        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(self.beam_size, -1) + scores.view(self.beam_size, 1)
        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(self.beam_size)
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // self.beam_size, best_k_idx_in_k2 % self.beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx
        return gen_seq, scores

    def translate_sentence(self, batch_data):
        src_ids = batch_data['source_ids']
        PE = batch_data['PE']

        assert src_ids.size(0) == 1
        # src_pos = self.encoder_PE

        with torch.no_grad():
            src_mask = self._get_pad_mask(src_ids)  # torch.Size([1, 1, 64])
            enc_output, gen_seq, scores, hidden_mat = self._get_init_state(src_ids, src_mask, PE)

            ans_idx = 0  # default
            for step in range(2, self.max_sen_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], hidden_mat, src_mask, PE)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == self.eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, self.max_sen_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** self.alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


#
#
#         print('using beam search: ', beam_search)
#
#     def _gen_id_seq(self, pred_tens):
#         batch_size = pred_tens.size(0)
#         pred_pro = F.softmax(pred_tens, dim=-1)
#         _, id_seq = pred_pro.topk(1, dim=-1, largest=True)
#         id_seq = id_seq.view(batch_size, self.max_sen_len).tolist()
#         id_seq = list(map(str, id_seq))
#         return id_seq
#
#     def _val(self, val_loader):
#         if self.beam_search:
#             for batch_data in val_loader:
#                 pred_tens = self.model(batch_data)
#
#
#         else:  # not beam search
#             for batch_data in val_loader:
#                 pred_tens = self.model(batch_data)  # [B, 64, 2000]
#                 id_seq = self._gen_id_seq(pred_tens)
#                 reference = [batch_data['target_id'].tolist()]
#                 bleu = sentence_bleu(reference, pred)


# class Translator():
#     def __init__(self, data_loader=None, beam_search=True, num_beams=5):
#         self.data_loader = data_loader
#         self.beam_search = beam_search
#         self.num_beams = num_beams
#
#     def _cal_BLEU_score(self, batch_pred_tens, batch_target_ids):
#         score = 0
#         return score
#
#     def forward(self, a):
#         return a
#
# translator = Translator()
# a = 1
# b = translator(a)
# print(b)


