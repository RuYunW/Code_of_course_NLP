import torch
from torch.utils.data import Dataset

class BatchData(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        text_info = {'input_ids': data['text_info']['input_ids'].to(self.device),
                     'token_type_ids': data['text_info']['token_type_ids'].to(self.device),
                     'attention_mask': data['text_info']['attention_mask'].to(self.device),
                     'val_token_type_ids': data['text_info']['val_token_type_ids'].to(self.device),
                     'tens_words': data['text_info']['tens_words'].to(self.device),
                     'tens_sentence': data['text_info']['tens_sentence'].to(self.device)}
        tag = data['tags'].to(self.device)

        return {'text_info': text_info, 'tags': tag}


