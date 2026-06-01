import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn as nn
import torch.optim as optim

class MappedDataset(Dataset):
    def __init__(self, subset, offset):
        self.subset = subset
        self.offset = offset
    def __getitem__(self, index):
        img, target = self.subset[index]
        return img, target - self.offset
    def __len__(self):
        return len(self.subset)

def get_dataloaders(config):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 采用 test 作为训练集 (6000+图像)，val 作为测试集
    full_train = datasets.Flowers102(root=config['data_dir'], split='test', download=True, transform=transform_train)
    full_test = datasets.Flowers102(root=config['data_dir'], split='val', download=True, transform=transform_test)
    
    targets_train = torch.tensor(full_train._labels)
    targets_test = torch.tensor(full_test._labels)
    
    priv_train_idx = (targets_train <= 50).nonzero(as_tuple=True)[0]
    priv_test_idx = (targets_test <= 50).nonzero(as_tuple=True)[0]
    pub_train_idx = (targets_train > 50).nonzero(as_tuple=True)[0]
    
    loader_priv_train = DataLoader(Subset(full_train, priv_train_idx), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    loader_priv_test = DataLoader(Subset(full_test, priv_test_idx), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    pub_subset = Subset(full_train, pub_train_idx)
    mapped_pub_subset = MappedDataset(pub_subset, offset=51)
    loader_pub_train = DataLoader(mapped_pub_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    
    return loader_priv_train, loader_priv_test, loader_pub_train

def evaluate(client, server, dataloader, device):
    client.eval()
    server.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = client(inputs)
            outputs = server(features)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def train_target_model(client, server, dataloader, config, device):
    optimizer = optim.Adam(list(client.parameters()) + list(server.parameters()), lr=config['target_lr'])
    criterion = nn.CrossEntropyLoss()
    
    print("  => 开始预训练 Target Model...")
    for epoch in range(config['target_epochs']):
        client.train()
        server.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = server(client(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_acc = 100. * correct / total
        print(f"     Epoch [{epoch+1}/{config['target_epochs']}] Loss: {running_loss/len(dataloader):.4f} | Train Acc: {epoch_acc:.2f}%")