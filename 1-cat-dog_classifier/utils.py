import os
import torch


def img_reader(path):
    data_list = []
    img_dir_list = os.listdir(path)
    for img_name in img_dir_list:
        img_path = path + img_name
        label = img_name.split('.')[0]
        data_list.append({'img_path': img_path, 'label': label})
    return data_list

def val(model, val_dataloader, criteria, batch_size):
    model.eval()
    with torch.no_grad():
        loss_all = []
        acc_all = []
        for data in val_dataloader:
            output = model(data)
            loss = criteria(output, data['label'])
            _, prediction = torch.max(output, 1)
            train_correct = (prediction == data['label']).sum()
            train_acc = train_correct.float() / batch_size
            loss_all.append(loss)
            acc_all.append(train_acc)

        loss = sum(loss_all) / len(loss_all)
        acc = sum(acc_all) / len(acc_all)

    model.train()
    return loss, acc


