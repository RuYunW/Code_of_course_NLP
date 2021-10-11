import torch
from torch.utils.data import  Dataset
from PIL import Image
import torchvision.transforms as transforms


class BatchData(Dataset):
    def __init__(self, all_data, img_size):
        self.all_data = all_data
        self.img_size = img_size

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]
        img_path = data['img_path']
        label = torch.tensor(0) if data['label'] == 'cat' else torch.tensor(1)  # cat = 1  dog = 0
        torch_img = self.transfor_img(img_path)
        img_info = {'img_path': img_path, 'tens': torch_img, 'label': label}
        return img_info

    def transfor_img(self, img_path):
        mode = Image.open(img_path).convert('RGB')
        input_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            #             transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        torch_img = input_transform(mode)
        mode.close()
        return torch_img