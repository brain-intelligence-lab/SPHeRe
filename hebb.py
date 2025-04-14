import os
import argparse
parser = argparse.ArgumentParser(description='Train a CNN with Hebbian learning on CIFAR-100')
parser.add_argument('--cuda', type=str, default='0', help='CUDA device (default: 0)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 2000)')
parser.add_argument('--dataset', type=str, default='cifar10', help='number of classes (default: 100)')
parser.add_argument('--is_bp', action='store_true', help='Use backpropagation (default: False)')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import logging, glob
from torchvision.io import read_image, ImageReadMode
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class TrainTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("./data/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("./data/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, image in enumerate(loader):
        inputs, labels = image
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, m_loss = model(inputs)
        loss = criterion(outputs, labels)
        loss += m_loss
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, image in enumerate(loader):
        inputs, labels = image
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, m_loss = model(inputs)
        loss = criterion(outputs, labels)
        loss += m_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total


class HebbLayer(nn.Module):
    def __init__(self, block, in_channel, out_channel, proj_num, is_bp, res=False):
        super(HebbLayer, self).__init__()
        self.block = block
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res = res
        self.criteria = nn.MSELoss()
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, 1, 1, 0),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channel // 2, proj_num),
        )
        self.is_bp = is_bp

    def normal_kernel(self, x0):
        x0 = x0.view(x0.size(0), -1)
        x = torch.norm(x0, dim=-1, keepdim=True, p=2)
        x = x + 1e-8 * (x <= 0).float()
        x0 = x0 / x
        cov = x0 @ x0.t()
        return cov
    
    def ortho_loss(self, x):
        x = x.view(x.size(0), -1)
        x = x.permute(1, 0)
        rx = self.normal_kernel(x)
        identity = torch.eye(rx.size(0), device=x.device)
        loss = F.mse_loss(rx, identity)
        return loss
    
    def hebb_loss(self, z, y):
        corr_y = self.normal_kernel(y)
        corr_z = self.normal_kernel(z)
        loss = self.criteria(corr_z, corr_y)
        return loss
    
    def cal_loss(self, x, y):
        y_proj = self.out_proj(y)
        unsup_hebb_loss = self.hebb_loss(x, y_proj)
        #unsup_hebb_img_loss = self.dis_loss(img, y_proj)
        ortho_loss = self.ortho_loss(y_proj)
        loss = unsup_hebb_loss + ortho_loss*0.8 #- unsup_hebb_img_loss * 0.3

        return loss

    def forward(self, x):
        y = self.block(x)
        loss = 0
        if self.training and not self.is_bp:
            loss = self.cal_loss(x, y)
        if self.res:
            if self.in_channel != self.out_channel:
                x = F.pad(x, (0, 0, 0, 0, 0, self.out_channel - self.in_channel))
                x = nn.AvgPool2d(2, 2)(x)
            y = y + x
        
        if not self.is_bp:
            y = y.detach()

        return y, loss

class CNN(nn.Module):
    def __init__(self, label_num=10, is_bp=None):
        super(CNN, self).__init__()
        self.proj_num = 256
        self.channel = 1536

        self.block1 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(3, self.channel // 4, 3, 1, 1),
                nn.BatchNorm2d(self.channel // 4),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            3, self.channel // 4, self.proj_num, is_bp, res=False
        )
        self.block2 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(self.channel // 4, self.channel // 2, 3, 1, 1),
                nn.BatchNorm2d(self.channel // 2),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            self.channel // 4, self.channel // 2, self.proj_num, is_bp, res=False
        )
        self.block3 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(self.channel // 2, self.channel, 3, 1, 1),
                nn.BatchNorm2d(self.channel),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            self.channel // 2, self.channel, self.proj_num, is_bp, res=True
        )
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(self.channel * 16, label_num)


    def forward(self, x):
        loss = 0
        x, loss1 = self.block1(x)
        loss += loss1

        x, loss2 = self.block2(x)
        loss += loss2

        x, loss3 = self.block3(x)
        loss += loss3
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, loss

if __name__ == '__main__':    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"unsup_log.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)
    
    seed_all(args.seed)
    batch_size = 128
    epochs = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'cifar100':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.48216, 0.44653), std=(0.247, 0.2434, 0.2616))
        ])
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.48216, 0.44653), std=(0.247, 0.2434, 0.2616))
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        label_num = 100
        
    if args.dataset == 'cifar10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.48216, 0.44653), std=(0.247, 0.2434, 0.2616))
        ])
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.48216, 0.44653), std=(0.247, 0.2434, 0.2616))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        label_num = 10
        
    
    if args.dataset == 'tiny-imagenet':
        id_dict = {}
        for i, line in enumerate(open('./data/tiny-imagenet-200/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        
        
        test_transform = transforms.Compose([
            transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
        ])
        train_transform = transforms.Compose([
            transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
        ])
        trainset = TrainTinyImageNetDataset(id = id_dict, transform = train_transform)
        testset = TestTinyImageNetDataset(id = id_dict, transform = test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        label_num = 200

    model = CNN(label_num, args.is_bp).to(device)
    print(args.is_bp)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    best_accu = 0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        scheduler.step()
        print(f'epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}')
        if test_acc > best_accu:
            best_accu = test_acc
    if args.is_bp:
        torch.save(model.state_dict(), f'./bp.pth')
    else:
        torch.save(model.state_dict(), f'./hebb.pth')
    logger.info(f'Best test accuracy: {best_accu}, last test accuracy: {test_acc}')