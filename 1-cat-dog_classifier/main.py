from dataset import BatchData
import torch
from torch.utils.data import  DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model import VGGNet16, SimpleCNN, SimpleDNN
from tqdm import tqdm
from utils import img_reader, val
import os

reshape_image_size = 224
batch_size = 20
num_steps = 500
num_samples = 2000
val_setp = 50
lr = 5e-5
img_size = 56

# 图片路径，相对路径
# cat_train_image_dir = "./data/train/cat."
# dog_train_image_dir = "./data/train/dog."
train_image_dir = './data/train/'
val_image_dir = './data/val/'

train_data_list = img_reader(train_image_dir)
val_data_list = img_reader(val_image_dir, 200)

train_dataset = BatchData(train_data_list, img_size)
val_dataset = BatchData(val_data_list, img_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# model = SimpleCNN()
model = SimpleDNN(input_dim=9408)
criteria = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
num_steps_per_epoch = num_samples // batch_size

model.train()
for step in tqdm(range(num_steps)):
    if step % num_steps_per_epoch == 0:
        iter_data = iter(train_dataloader)
    data = iter_data.next()
    output = model(data)
    label = data['label']
    loss = criteria(output, label)

    _, prediction = torch.max(output, 1)
    train_correct = (prediction == data['label']).sum()
    train_acc = train_correct.float() / batch_size

    # print('loss = {:.2f}, acc = {:.2f}'.format(loss, train_acc))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % val_setp == 0:
        val_loss, val_acc = val(model, val_dataloader, criteria, batch_size)
        print('val_loss = {:.2f}, val_acc = {:.2f}'.format(val_loss, val_acc))
