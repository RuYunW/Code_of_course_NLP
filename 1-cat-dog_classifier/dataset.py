import torch
from torch.utils.data import  Dataset
from PIL import Image
import torchvision.transforms as transforms

class BatchData(Dataset):
    def __init__(self, all_data, img_size=112, usingRNN=False, patch_size=38):
        self.all_data = all_data
        self.img_size = img_size
        self.usingRNN = usingRNN

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
            transforms.ToTensor(),
        ])
        torch_img = input_transform(mode)
        mode.close()
        if self.usingRNN:  # 3*3 patch
            patch_size = int(self.img_size / 3)
            img_patch_tens = torch.zeros((9, 3, patch_size, patch_size))

            for i in range(3):
                for j in range(3):
                    img_patch_tens[i*3+j] = torch_img[:, i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
            return img_patch_tens
        else:
            return torch_img