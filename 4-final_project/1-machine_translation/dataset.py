from torch.utils.data import Dataset
import torch

class BatchData(Dataset):
    def __init__(self, source_ids, target_ids, source_vocab_len, target_vocab_len, max_sen_len, source_token2id, target_token2id):
        assert len(source_ids) == len(target_ids), 'The length of source data and target data are not equal to the target data. '
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_vocab_len = source_vocab_len
        self.target_vocab_len = target_vocab_len
        self.max_sen_len = max_sen_len
        self.source_token2id = source_token2id
        self.target_token2id = target_token2id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.source_ids)

    def _get_mask(self, length):
        mask = torch.ones(self.max_sen_len)
        mask[length:] = 0
        return mask

    def __getitem__(self, item):
        # source
        source_ids = self.source_ids[item]
        source_mask_tens = self._get_mask(len(source_ids))
        if len(source_ids) > self.max_sen_len:
            source_ids = source_ids[:self.max_sen_len]
        else:
            source_ids = source_ids+[self.source_token2id['<PAD>'] for _ in range(self.max_sen_len-len(source_ids))]
        # # onehot
        # source_tens = torch.zeros([self.max_sen_len, self.source_vocab_len]).long()
        # for i, id in enumerate(source_ids):
        #     source_tens[i][id] = 1.

        # target
        target_ids = [self.target_token2id['<SOS>']] + self.target_ids[item] + [self.target_token2id['<EOS>']]
        target_mask_tens = self._get_mask(len(target_ids))
        if len(target_ids) > self.max_sen_len:
            target_ids = target_ids[:self.max_sen_len]
        else:
            target_ids = target_ids+[self.target_token2id['<PAD>'] for _ in range(self.max_sen_len-len(target_ids))]
        # # onehot
        # target_tens = torch.zeros([self.max_sen_len, self.target_vocab_len]).long()
        # for i, id in enumerate(target_ids):
        #     target_tens[i][id] = 1.

        data_info = {'source_ids': torch.tensor(source_ids).to(self.device),
                     'target_ids': torch.tensor(target_ids).to(self.device),
                     'source_mask_tens': source_mask_tens.to(self.device),
                     'target_mask_tens': target_mask_tens.to(self.device),
                     }

        return data_info
