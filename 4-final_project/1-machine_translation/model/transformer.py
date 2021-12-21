import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def positional_embedding(max_sen_len=64, d_model=512):
    PE = torch.zeros([max_sen_len, d_model]).to(device)
    for pos in range(max_sen_len):
        for i in range(int(d_model / 2)):
            PE[pos][2*i]   = math.sin(pos / (10000 ** ((2*i) / d_model)))
            PE[pos][2*i+1] = math.cos(pos / (10000 ** ((2*i) / d_model)))
    return PE

class Transformer(nn.Module):
    def __init__(self, num_enc_layers=6, num_dec_layers=6, max_sen_len=64, d_model=512,
                 input_size=2000+4, output_size=2000+4, pad_idx=2, dropout=0.1, num_heads=8, d_inner=2048):
        super(Transformer, self).__init__()
        # self.num_enc_layers = num_enc_layers
        # self.num_dec_layers = num_dec_layers
        # self.max_sen_len = max_sen_len
        # self.d_model = d_model
        # self.input_size = input_size
        # self.output_size = output_size
        self.pad_idx = pad_idx
        self.PE = positional_embedding()
        # self.dropout = dropout
        # self.is_eval = is_eval
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # Networks
        self.src_embedding = nn.Embedding(input_size, d_model)
        self.tgt_embedding = nn.Embedding(output_size, d_model)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_inner=d_inner, num_heads=num_heads, max_sen_len=max_sen_len,
                         dropout=dropout) for _ in range(num_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner=d_inner, max_sen_len=max_sen_len,
                         dropout=dropout) for _ in range(num_dec_layers)
        ])
        self.linear = nn.Linear(d_model, output_size)
        # self.linear = nn.Sequential(
        #     nn.Linear(d_model, 2048),
        #     nn.Linear(2048, output_size)
        # )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

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
        # src_ids = data['source_ids']
        # tgt_ids = data['target_ids']
        # PE = data['PE']

        # Encoder Embedding
        src_emb = self.src_embedding(src_ids)
        src_pos = self.PE
        src_feats = src_emb + src_pos
        src_feats = self.dropout(src_feats)
        src_feats = self.layer_norm(src_feats)
        # Decoder Embedding
        tgt_emb = self.tgt_embedding(tgt_ids)
        tgt_pos = self.PE
        tgt_feats = tgt_emb + tgt_pos
        tgt_feats = self.dropout(tgt_feats)
        tgt_feats = self.layer_norm(tgt_feats)
        # Mask
        src_mask = self._get_pad_mask(src_ids)
        tgt_mask = self._get_pad_mask(tgt_ids) & self._get_subsequent_mask(tgt_ids)
        # Encoder
        for encoder_block in self.encoder:
            src_feats, hidden_mat = encoder_block(src_feats, src_mask)
        # Decoder
        for decoder_block in self.decoder:
            tgt_feats = decoder_block(tgt_feats, hidden_mat, tgt_mask, src_mask)

        tgt_feats = self.linear(tgt_feats)
        return tgt_feats.view(-1, tgt_feats.size(2))


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, dropout=0.1, num_heads=8, max_sen_len=64):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model

        # networks
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_sen_len=max_sen_len, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(self.d_model, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, self.d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_tens, src_mask):
        # Multi_head attention
        input_tens = self.dropout(input_tens)
        input_tens = self.layer_norm(input_tens)
        tens, hidden_mat = self.multi_head_attention(input_tens, src_mask)
        tens = self.linear(tens)
        tens = self.dropout(tens)
        tens = self.layer_norm(tens + input_tens)

        # Feedforward networks
        tens = self.feedforward(tens)
        tens = self.dropout(tens)
        tens = self.layer_norm(input_tens + tens)
        # tens = self.layer_norm(tens + self.fc(tens))
        return tens, hidden_mat


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_inner=2048, num_heads=8, max_sen_len=64, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.max_len_sen = max_sen_len
        # self.is_eval = is_eval

        # network
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model=d_model, num_heads=num_heads, max_sen_len=max_sen_len, dropout=dropout)
        self.linear_1 = nn.Linear(self.d_model, self.d_model)
        self.linear_2 = nn.Linear(self.d_model, self.d_model)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_sen_len=max_sen_len, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Linear(d_inner, d_model)
        )
        self.norm_layer = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden_mat, slf_attn_mask=None, dec_enc_attn_mask=None):
        tens_1 = self.masked_multi_head_attention(dec_input, slf_attn_mask)
        tens_1 = self.linear_1(tens_1)
        tens_1 = self.dropout(tens_1)
        tens_1 = self.norm_layer(dec_input + tens_1)

        tens_2, _ = self.multi_head_attention(tens_1, dec_enc_attn_mask, hidden_mat)
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # networks
        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_heads * self.d_v, bias=False)
        self.fc = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def _scaled_dot_product_attention(self, Q, K, V, mask):
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(Q, K.transpose(2, 3)) / scale
        attn = attn.masked_fill(mask == 0, -1e9)
        Z = torch.matmul(F.softmax(attn, dim=-1), V)
        return Z

    def forward(self, input_tens, mask, hidden_mat=None):
        batch_size = input_tens.size(0)
        len_q = input_tens.size(1)
        len_k = input_tens.size(1)
        len_v = input_tens.size(1)

        Q = self.W_Q(input_tens).view(batch_size, len_q, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)

        if not hidden_mat:  # encoder
            K = self.W_K(input_tens).view(batch_size, len_k, self.num_heads, self.d_k)
            V = self.W_V(input_tens).view(batch_size, len_v, self.num_heads, self.d_v)
            K, V = K.transpose(1, 2), V.transpose(1, 2)
        else:  # decoder
            _, K, V = hidden_mat

        hidden_mat = (Q, K, V)

        mask = mask.unsqueeze(1)
        Z = self._scaled_dot_product_attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        Z = self.dropout(self.fc(Z))
        return Z, hidden_mat


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512, max_sen_len=64, dropout=0.1):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_sen_len = max_sen_len
        # self.dropout = dropout
        # self.is_eval = is_eval
        self.d_q = int(d_model / num_heads)
        self.d_k = int(d_model / num_heads)
        self.d_v = int(d_model / num_heads)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # networks
        self.W_Q = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_model, self.num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_model, self.num_heads * self.d_v, bias=False)
        self.fc = nn.Linear(self.num_heads * self.d_v, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, Q, K, V, mask):
        attn = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(2, 3))
        attn = attn.masked_fill(mask == 0, -1e9)
        Z = torch.matmul(F.softmax(attn, dim=-1), V)
        return Z

    def forward(self, input_tens, mask):
        batch_size = input_tens.size(0)
        len_q = input_tens.size(1)
        len_k = input_tens.size(1)
        len_v = input_tens.size(1)

        # print(input_tens.shape)  # torch.Size([1, 1, 512])
        # exit()
        Q = self.W_Q(input_tens).view(batch_size, len_q, self.num_heads, self.d_k)
        K = self.W_K(input_tens).view(batch_size, len_k, self.num_heads, self.d_k)
        V = self.W_V(input_tens).view(batch_size, len_v, self.num_heads, self.d_v)

        # Transpose for attention dot product: b * h * L * dv
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        mask = mask.unsqueeze(1)
        Z = self._scaled_dot_product_attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        Z = self.dropout(self.fc(Z))
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