import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class PostionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=64, device=device):
        super(PostionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class Embedder(nn.Module):
    def __init__(self, input_dim, d_model, max_sen_len, dropout):
        super(Embedder, self).__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(input_dim, d_model)
        self.pos = PostionalEncoding(d_model, max_sen_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, ids):
        emb = self.emb(ids)
        # emb *= self.d_model ** 0.5
        pos = self.pos(ids)
        feat = emb + pos
        feat = self.dropout(feat)
        feat = self.layer_norm(feat)
        return feat


class Transformer(nn.Module):
    def __init__(self, num_enc_layers=6, num_dec_layers=6, max_sen_len=64, d_model=512,
                 input_size=2000+4, output_size=2000+4, pad_idx=2, dropout=0.1, num_heads=8, d_inner=2048,
                 trg_emb_prj_weight_sharing=True):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.scale_prj = True
        self.encoder = Encoder(input_dim=input_size, num_layers=num_enc_layers, d_model=d_model, d_inner=d_inner,
                               num_heads=num_heads, dropout=dropout, max_sen_len=max_sen_len)
        self.decoder = Decoder(output_dim=output_size, num_layers=num_dec_layers, d_model=d_model, d_inner=d_inner,
                               num_heads=num_heads, dropout=dropout, max_sen_len=max_sen_len)
        self.linear = nn.Linear(d_model, output_size)
        if trg_emb_prj_weight_sharing:
            self.linear.weight = self.decoder.embedder.emb.weight

    def _get_pad_mask(self, ids):
        mask = (ids != self.pad_idx).unsqueeze(-2)
        return mask

    def _get_subsequent_mask(self, ids):
        ''' For masking out the subsequent info. '''
        sz_b, len_s = ids.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=ids.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(self, src_ids, tgt_ids):
        # Mask
        src_mask = self._get_pad_mask(src_ids)  # [16, 1, 64]
        tgt_mask = self._get_pad_mask(tgt_ids) & self._get_subsequent_mask(tgt_ids)  # [16, 64, 64]

        enc_output = self.encoder(src_ids, src_mask)
        dec_output = self.decoder(tgt_ids, enc_output, tgt_mask, src_mask)

        tgt_feats = self.linear(dec_output)  # [B, 64, 512]
        tgt_feats *= self.d_model ** -0.5
        return tgt_feats.view(-1, tgt_feats.size(2))

class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers=6, d_model=512, d_inner=2048, dropout=0.1, num_heads=8, max_sen_len=64):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.scale_emb = False

        self.embedder = Embedder(input_dim, d_model, max_sen_len, dropout)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_inner=d_inner, num_heads=num_heads, max_sen_len=max_sen_len,
                         dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, src_ids, src_mask):
        enc_output = self.embedder(src_ids)
        # if self.scale_emb:
        #     enc_output *= self.d_model ** 0.5
        for enc_blk in self.encoder:
            enc_output = enc_blk(enc_output, src_mask)

        return enc_output


class Decoder(nn.Module):
    def __init__(self, output_dim, num_layers=6, d_model=512, d_inner=2048, dropout=0.1, num_heads=8, max_sen_len=64):
        super(Decoder, self).__init__()
        self.d_model = d_model
        # self.scale_emb = False

        self.embedder = Embedder(output_dim, d_model, max_sen_len, dropout)
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner=d_inner, num_heads=num_heads, max_sen_len=max_sen_len,
                         dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, src_ids, enc_output, tgt_mask, src_mask):
        dec_output = self.embedder(src_ids)
        # if self.scale_emb:
        #     dec_output *= self.d_model ** 0.5

        for dec_blk in self.decoder:
            dec_output = dec_blk(dec_output, enc_output, tgt_mask, src_mask)

        return dec_output


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, dropout=0.1, num_heads=8, max_sen_len=64):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model

        # networks
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads,
                                                       max_sen_len=max_sen_len, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(self.d_model, d_inner),
            nn.LeakyReLU(),
            nn.Linear(d_inner, self.d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_tens, src_mask):
        enc_output = enc_tens
        # enc_output *= self.d_model ** 0.5
        tens = self.multi_head_attention(enc_output, enc_output, enc_output, src_mask)

        tens = self.linear(tens)
        tens = self.dropout(tens)
        tens = self.layer_norm(tens + enc_output)

        # Feedforward networks
        tens = self.feedforward(tens)
        tens = self.dropout(tens)
        tens = self.layer_norm(enc_output + tens)
        return tens


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, num_heads=8, max_sen_len=64, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.max_len_sen = max_sen_len

        # network
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_sen_len=max_sen_len, dropout=dropout)
        self.linear_1 = nn.Linear(self.d_model, self.d_model)
        self.linear_2 = nn.Linear(self.d_model, self.d_model)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_sen_len=max_sen_len, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.LeakyReLU(),
            nn.Linear(d_inner, d_model)
        )
        self.norm_layer = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        tens_1 = self.masked_multi_head_attention(dec_input, dec_input, dec_input, slf_attn_mask)  # [16, 64, 512]
        tens_1 = self.linear_1(tens_1)
        tens_1 = self.dropout(tens_1)
        tens_1 = self.norm_layer(dec_input + tens_1)

        tens_2 = self.multi_head_attention(tens_1, enc_output, enc_output, dec_enc_attn_mask)
        tens_2 = self.linear_2(tens_2)
        tens_2 = self.dropout(tens_2)
        tens_2 = self.norm_layer(tens_1 + tens_2)

        tens_3 = self.feedforward(tens_2)
        tens_3 = self.dropout(tens_3)
        tens_3 = self.norm_layer(tens_2 + tens_3)

        return tens_3


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512, max_sen_len=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_sen_len = max_sen_len
        self.d_q = int(d_model / num_heads)
        self.d_k = int(d_model / num_heads)
        self.d_v = int(d_model / num_heads)

        # networks
        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_heads * self.d_v, bias=False)
        self.fc = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def _scaled_dot_product_attention(self, Q, K, V, mask):
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(Q/scale, K.transpose(2, 3))
        attn = attn.masked_fill(mask == 0, -1e9)
        Z = torch.matmul(F.softmax(attn, dim=-1), V)
        return Z

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key.size(1)
        len_v = value.size(1)

        Q = self.W_Q(query).view(batch_size, len_q, self.num_heads, self.d_k)
        K = self.W_K(key).view(batch_size, len_k, self.num_heads, self.d_k)
        V = self.W_V(value).view(batch_size, len_v, self.num_heads, self.d_v)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        mask = mask.unsqueeze(1)
        Z = self._scaled_dot_product_attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        return Z


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr