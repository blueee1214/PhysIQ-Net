import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from pretrained_syreanet import *
from torch.utils.data import ConcatDataset
from iqanet import *
from pretrained_retinex.decomnet import *

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, '\n')


# settings ==========================================================================================
data_path = '/root/autodl-tmp/jiaxue/SAUD2.0'
models_save_path = './models_save/'
pretrained_model_path = os.path.join(models_save_path,'/scau/models_save/pretrained_model_24.pth')
mos_file = os.path.join(data_path, 'mos_result/mos.xlsx')
results_file = os.path.join(data_path, 'mos_result/result.xlsx')
record_file = os.path.join(data_path, 'mos_result/record.txt')
train = True
continue_train = False
test_ratio = 0.2
random_state = 42
batch_size = 4
num_workers = 8
num_epochs = 100
milestones = []
test_interval = 1
learning_rate = 0.0002
weight_decay = 0.00005
# ==================================================================================================

syreanet = SyreaNet().to(device)
syreanet.load('/scau/backup_disks/PhysIQ-Net/pretrained_syreanet/pretrained_syreanet.pth')
decomnet = DecomNet().to(device)
ckpt = torch.load('/scau/backup_disks//PhysIQ-Net/pretrained_Retinex/pretrained_Retinex.tar')
decomnet.load('/scau/backup_disks//PhysIQ-Net/pretrained_Retinex/pretrained_Retinex.tar')
for param in syreanet.parameters():
    param.requires_grad = False
for param in decomnet.parameters():
    param.requires_grad = False
model = IQANet().to(device)


if train:
    if not continue_train:
        train_dataset, test_dataset = split_dataset(mos_file, data_path, test_size=test_ratio, random_state=random_state)
        save_dataset(train_dataset, test_dataset, data_path)
        print("Training From Scratch!\n")
    else:
        train_dataset, test_dataset = load_dataset(data_path)
        model.load_state_dict(torch.load(pretrained_model_path))
        print("Load Pretrained Model!\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    criterion = nn.MSELoss()

    optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.1, last_epoch=-1)

    train_test_model(model, syreanet, decomnet, train_loader, test_loader, num_epochs, test_interval, device, criterion, optimizer, scheduler, results_file, record_file, models_save_path)

else:
    model.load_state_dict(torch.load(pretrained_model_path))
    print("Load Pretrained Model!\n")

    train_dataset, test_dataset = load_dataset(data_path)
    train_dataset.train = False
    overall_dataset = ConcatDataset([train_dataset, test_dataset])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_model(model, syreanet, decomnet, train_loader, device, results_file, record_file)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_model(model, syreanet, decomnet, test_loader, device, results_file, record_file)

    overall_loader = DataLoader(overall_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_model(model, syreanet, decomnet, overall_loader, device, results_file, record_file)
