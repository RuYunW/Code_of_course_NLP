import torch
import torch.nn as nn
import math

src_pad_idx = 2
trg_pad_idx = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def make_src_mask(src):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_trg_mask(trg):
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).bool().to(device)
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask


class PostionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
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


class ScaleDotProductAttention(nn.Module):
    def __init__(self, n_head=8):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()
        self.n_head = n_head

    def forward(self, q, k, v, mask=None, e=1e-12):
        print(k.size())
        batch_size, head, length, d_tensor = k.size()
        # _, length, d_tensor = k.size()




        # 1. dot product Query with Key^T to compute similarity
        k_t = k.view(batch_size, head, d_tensor, length)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.model_dim=d_model
        self.n_head=n_head
        self.head_dim = self.model_dim // self.n_head

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

        self.linear_final=nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaleDotProductAttention(n_head)


    def forward(self, query, key, value, mask=None):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size=k.size()[0]

        q_ = q.view(batch_size * self.n_head, -1, self.head_dim)
        k_ = k.view(batch_size * self.n_head, -1, self.head_dim)
        v_ = v.view(batch_size * self.n_head, -1, self.head_dim)

        context = self.scaled_dot_product_attention(q_, k_, v_, mask)

        output = context.view(batch_size, -1, self.head_dim * self.n_head)
        output = self.linear_final(output)
        output = self.dropout(output)
        return output

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout_rate=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        _x = x
        x = self.attention(x, x, x, mask=s_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout_rate=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout_rate=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.self_attention(dec, dec, dec, mask=t_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(x, enc, enc, mask=s_mask)
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        _x = x
        x = self.ffn(x)
        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, drop_prob, max_len, vocab_size, device):
        super(TransformerEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PostionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, trg_ids):
        emb = self.emb(trg_ids)
        pos = self.pos_enc(trg_ids)
        feat = emb + pos
        feat = self.dropout(feat)
        feat = self.layer_norm(feat)
        return feat


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)

        return output

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)


    def forward(self, src, trg):
        src_mask = make_src_mask(src)
        trg_mask = make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output



