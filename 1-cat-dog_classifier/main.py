from dataset import BatchData
import torch
from torch.utils.data import  DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model import SimpleCNN, SimpleDNN, SimplePatchRNN
from tqdm import tqdm
from utils import img_reader, val
import matplotlib.pyplot as plt


batch_size = 20
num_steps = 1000
val_setp = 50
lr = 0.01
network = 'CNN'

if network == 'CNN':
    img_size = 112
elif network == 'RNN':
    img_size = 114  # 48*3
else:  # DNN
    img_size = 56

# 图片路径，相对路径
train_image_dir = './data/train/'
val_image_dir = './data/val/'

train_data_list = img_reader(train_image_dir)
val_data_list = img_reader(val_image_dir)

train_dataset = BatchData(train_data_list, img_size, usingRNN=True if network == 'RNN' else False)
val_dataset = BatchData(val_data_list, img_size, usingRNN=True if network == 'RNN' else False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

if network == 'CNN':
    model = SimpleCNN()
elif network == 'RNN':
    model = SimplePatchRNN(batch_size=batch_size)
else:
    model = SimpleDNN(input_dim=img_size*img_size*3)


criteria = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
# optimizer = SGD(model.parameters(), lr=lr)
num_steps_per_epoch = len(train_data_list) // batch_size

model.train()
val_loss_list = []
val_acc_list = []
step_list = []
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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % val_setp == 0 or step == num_steps - 1:
        val_loss, val_acc = val(model, val_dataloader, criteria, batch_size)
        val_loss_list.append(round(float(val_loss), 2))
        val_acc_list.append(round(float(val_acc), 2))
        step_list.append(step)
        print('val_loss = {:.2f}, val_acc = {:.2f}'.format(val_loss, val_acc))

fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax1 = fig.add_subplot(111)
ax1.plot(step_list, val_loss_list, 'b')
ax1.set_ylabel('loss value', fontsize=12)
plt.legend(['loss'], loc='upper left')
ax2 = ax1.twinx()
ax2.plot(step_list, val_acc_list, 'r')
ax2.legend(['acc'], loc='upper right')
ax2.set_ylabel('accuracy value', fontsize=12)
plt.title(network + '神经网络val集loss值与acc值在训练中变化情况')
plt.xlabel('迭代步数')
plt.show()
