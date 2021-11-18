from transformers import BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
from model.CRF import BatchCRF

class BertLM(nn.Module):
    def __init__(self, max_len=64, base_model='bert-large-cased', hidden_dim=1024, num_hidden_layers=6):
        super(BertLM, self).__init__()
        # self.batch_size = batch_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.tag_to_ix = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
        self.tagset_size = len(self.tag_to_ix)

        self.bert_config = BertConfig.from_pretrained(base_model)
        # self.bert_config.type_vocab_size = 3
        self.bert_config.num_hidden_layers = num_hidden_layers  # 层数
        #         self.bert_config.num_attention_heads=16
        self.bert_config.max_position_embeddings = max_len


        self.bert_modle = BertModel(self.bert_config)
        self.bert2tag_fc = nn.Linear(self.hidden_dim, self.tagset_size+2)
        self.crf = BatchCRF(self.tag_to_ix)
        # self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # self.crf_layer = CRF(self.tag_to_ix)

    def loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        loss_value = self.crf._neg_log_likelihood_loss(feats, mask, tags)
        loss_value /= float(batch_size)
        return loss_value

    # def _feat_softmax(self, feat):
    #     feat_softmax = F.softmax(feat, dim=0)
    #     return feat_softmax

    def forward(self, data):
        text_info = data['text_info']
        # sentence_len = data['sentence_len']

        text_f = text_info['tens_words']
        text_t = text_info['token_type_ids']
        text_m = text_info['attention_mask']

        # input sentence into BERT
        output_bert = self.bert_modle(inputs_embeds=text_f,
                                      token_type_ids=text_t.long(),
                                      attention_mask=text_m.long())
        output_tens = output_bert[0]
        # reflecting features into tags
        fc_feat = self.bert2tag_fc(output_tens)
        # trans_feat = self.transitions(fc_feat)

        return fc_feat