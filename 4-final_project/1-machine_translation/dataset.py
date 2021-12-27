from torch.utils.data import Dataset
import torch
# from utils.utils import positional_embedding

class BatchData(Dataset):
    def __init__(self, source_ids, target_ids,  max_sen_len):
        assert len(source_ids) == len(target_ids), 'The length of source data and target data are not equal to the target data. '
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.max_sen_len = max_sen_len
        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, item):
        # source
        source_ids = [self.sos_id] + self.source_ids[item] + [self.eos_id]
        if len(source_ids) > self.max_sen_len:
            source_ids = source_ids[:self.max_sen_len]
        else:
            source_pad_ids = [self.pad_id] * (self.max_sen_len-len(source_ids))
            source_ids = source_ids + source_pad_ids

        # target
        target_ids = [self.sos_id] + self.target_ids[item] + [self.eos_id]
        if len(target_ids) > self.max_sen_len:
            target_ids = target_ids[:self.max_sen_len]
        else:
            target_pad_ids = [self.pad_id] * (self.max_sen_len - len(target_ids))
            target_ids = target_ids + target_pad_ids

        data_info = {'source_ids': torch.tensor(source_ids).to(self.device),
                     'target_ids': torch.tensor(target_ids).to(self.device)}

        return data_info
