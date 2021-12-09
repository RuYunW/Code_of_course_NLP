import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, num_enc_layers=6, num_dec_layers=6, max_sen_len=64, hidden_size=512,
                 input_size=2000, output_size=2000, dropout=0.1, is_eval=False):
        super(Transformer, self).__init__()
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.max_sen_len = max_sen_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.is_eval = is_eval
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.src_embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.tgt_embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.encoder = nn.ModuleList([EncoderBlock() for _ in range(self.num_enc_layers)])
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(self.num_dec_layers)])
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.Linear(2048, self.output_size)
        )
        self.dropout = nn.Dropout(self.dropout)

    def _positional_embedding(self):
        PE = torch.zeros([self.max_sen_len, self.hidden_size])
        for pos in range(self.max_sen_len):
            for i in range(int(self.hidden_size / 2)):
                PE[pos][2*i]   = math.sin(pos / (10000**((2*i)/self.hidden_size)))
                PE[pos][2*i+1] = math.cos(pos / (10000**((2*i)/self.hidden_size)))
        return PE.to(self.device)

    def _get_pad_mask(self, mask):
        return (mask != 0).unsqueeze(-2).to(self.device)

    def _get_subsequent_mask(self, mask):
        subsequent_mask = (1 - torch.triu(torch.ones((1, self.max_sen_len, self.max_sen_len), device=mask.device), diagonal=1)).bool()
        return subsequent_mask.to(self.device)

    def forward(self, data):
        src_ids = data['source_ids']
        tgt_ids = data['target_ids']
        src_mask = data['source_mask_tens']
        tgt_mask = data['target_mask_tens']
        # Encoder Embedding
        src_emb = self.src_embedding(src_ids)
        src_pos = self._positional_embedding()
        src_feats = src_emb + src_pos
        # Decoder Embedding
        tgt_emb = self.tgt_embedding(tgt_ids)
        tgt_pos = self._positional_embedding()
        tgt_feats = tgt_emb + tgt_pos

        src_mask = self._get_pad_mask(src_mask)
        tgt_mask = self._get_pad_mask(tgt_mask) & self._get_subsequent_mask(tgt_mask)

        for encoder_block in self.encoder:
            src_feats, hidden_mat = encoder_block(src_feats, src_mask)
        for decoder_block in self.decoder:
            tgt_feats = decoder_block(tgt_feats, tgt_mask, hidden_mat)

        return tgt_feats


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim=512, is_last=False):
        super(EncoderBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # networks
        self.multi_head_attention = MultiHeadAttention()
        self.norm_layer = nn.LayerNorm(self.hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.Linear(2048, self.hidden_dim)
        )

    def forward(self, input_tens, src_mask):
        tens, hidden_mat = self.multi_head_attention(input_tens, src_mask)
        tens = self.norm_layer(input_tens + tens)
        tens = self.norm_layer(tens + self.fc(tens))
        return tens, hidden_mat


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim=512, max_sen_len=64, is_eval=False):
        super(DecoderBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len_sen = max_sen_len
        self.is_eval = is_eval

        # network
        self.masked_multi_head_attention = MaskedMultiHeadAttention()
        self.norm_layer = nn.LayerNorm(self.hidden_dim)
        self.multi_head_attention = MultiHeadAttention()
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, decoder_input, mask, hidden_mat):
        tens_1 = self.masked_multi_head_attention(decoder_input, mask)
        tens_1 = self.norm_layer(decoder_input + tens_1)

        tens_2, _ = self.multi_head_attention(tens_1, mask, hidden_mat)
        tens_2 = self.norm_layer(tens_1 + tens_2)

        tens_3 = self.fc(tens_2)
        tens_3 = self.norm_layer(tens_2 + tens_3)

        return tens_3


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, max_sen_len=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_sen_len = max_sen_len
        self.d_q = int(hidden_dim / num_heads)
        self.d_k = int(hidden_dim / num_heads)
        self.d_v = int(hidden_dim / num_heads)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # networks
        self.W_Q = nn.Linear(self.hidden_dim, self.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.hidden_dim, self.num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.hidden_dim, self.num_heads * self.d_v, bias=False)
        self.fc = nn.Linear(self.num_heads * self.d_v, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, Q, K, V, mask):
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(Q, K.transpose(2, 3)) / scale
        attn = attn.masked_fill(mask == 0, -1e9)
        Z = torch.matmul(F.softmax(attn, dim=-1), V)
        return Z

    def forward(self, input_tens, mask, hidden_mat=None):
        batch_size = input_tens.size(0)
        Q = self.W_Q(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)

        if not hidden_mat:  # encoder
            K = self.W_K(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_k)
            V = self.W_V(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_v)
            K, V = K.transpose(1, 2), V.transpose(1, 2)
        else:  # decoder
            _, K, V = hidden_mat

        hidden_mat = (Q, K, V)

        mask = mask.unsqueeze(1)
        Z = self._scaled_dot_product_attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, self.max_sen_len, -1)
        Z = self.dropout(self.fc(Z))
        return Z, hidden_mat


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, max_sen_len=64, dropout=0.1, is_eval=False):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_sen_len = max_sen_len
        # self.dropout = dropout
        self.is_eval = is_eval
        self.d_q = int(hidden_dim / num_heads)
        self.d_k = int(hidden_dim / num_heads)
        self.d_v = int(hidden_dim / num_heads)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # networks
        self.W_Q = nn.Linear(self.hidden_dim, self.num_heads*self.d_k, bias=False)
        self.W_K = nn.Linear(self.hidden_dim, self.num_heads*self.d_k, bias=False)
        self.W_V = nn.Linear(self.hidden_dim, self.num_heads*self.d_v, bias=False)
        self.fc = nn.Linear(self.num_heads*self.d_v, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, Q, K, V, mask):
        attn = torch.matmul(Q / math.sqrt(self.d_k), K.transpose(2, 3))
        attn = attn.masked_fill(mask == 0, -1e9)
        Z = torch.matmul(F.softmax(attn, dim=-1), V)
        return Z

    def forward(self, input_tens, mask):
        batch_size = input_tens.size(0)

        Q = self.W_Q(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_k)
        K = self.W_K(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_k)
        V = self.W_V(input_tens).view(batch_size, self.max_sen_len, self.num_heads, self.d_v)

        # Transpose for attention dot product: b * h * L * dv
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        mask = mask.unsqueeze(1)
        Z = self._scaled_dot_product_attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, self.max_sen_len, -1)
        Z = self.dropout(self.fc(Z))
        return Z
