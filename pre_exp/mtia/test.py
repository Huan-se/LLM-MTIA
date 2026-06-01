import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset


# ==========================================
# 0. 数据集加载与预处理 (包含标签重映射机制)
# ==========================================
# 新增：用于对子集标签进行平移映射的 Dataset 包装器
class MappedDataset(Dataset):
    def __init__(self, subset, offset):
        self.subset = subset
        self.offset = offset

    def __getitem__(self, index):
        img, target = self.subset[index]
        # 将原始标签平移，防止交叉熵损失越界
        return img, target - self.offset

    def __len__(self):
        return len(self.subset)


def get_dataloaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("正在加载 CIFAR-100 数据集...")
    full_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    full_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    targets_train = torch.tensor(full_train.targets)
    targets_test = torch.tensor(full_test.targets)

    # 目标私有数据集 D_priv: 类别 0-49 (50个类)
    priv_train_idx = (targets_train < 50).nonzero(as_tuple=True)[0]
    priv_test_idx = (targets_test < 50).nonzero(as_tuple=True)[0]

    # 攻击者公共数据集 D_pub: 类别 50-99 (50个类)
    pub_train_idx = (targets_train >= 50).nonzero(as_tuple=True)[0]

    loader_priv_train = DataLoader(Subset(full_train, priv_train_idx), batch_size=batch_size, shuffle=True)
    loader_priv_test = DataLoader(Subset(full_test, priv_test_idx), batch_size=batch_size, shuffle=False)

    # 修复核心：将公共数据集的标签从 50~99 映射到 0~49
    pub_subset = Subset(full_train, pub_train_idx)
    mapped_pub_subset = MappedDataset(pub_subset, offset=50)
    loader_pub_train = DataLoader(mapped_pub_subset, batch_size=batch_size, shuffle=True)

    return loader_priv_train, loader_priv_test, loader_pub_train, 50, 50


# ==========================================
# 1. 网络定义与拆分 (使用标准的 ResNet-50)
# ==========================================
class ClientModel_True(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer1(x)


class ServerModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x, return_feature=False):
        x = self.layer2(x)
        feature_deep = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        if return_feature:
            return out, feature_deep
        return out


class SubstituteClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x, return_all=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        if return_all:
            return out1, out2
        return out2


# ==========================================
# 2. 核心辅助函数 (训练、测试与SAD损失)
# ==========================================
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


def get_attention_map(A):
    G = torch.sum(torch.abs(A) ** 2, dim=1, keepdim=True)
    B, _, H, W = G.size()
    G_flat = G.view(B, -1)
    Phi_flat = F.softmax(G_flat, dim=1)
    return Phi_flat.view(B, 1, H, W)


def sad_loss(A_shallow, A_deep):
    phi_shallow = get_attention_map(A_shallow)
    phi_deep = get_attention_map(A_deep)
    phi_deep_up = F.interpolate(phi_deep, size=phi_shallow.shape[2:], mode='bilinear', align_corners=False)
    return torch.norm(phi_shallow - phi_deep_up, p=2)


# ==========================================
# 3. 主实验流程
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader_priv_train, loader_priv_test, loader_pub_train, num_priv_classes, num_pub_classes = get_dataloaders(
        batch_size=64)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Phase 1: 训练完整的目标模型 (ResNet-50 on Priv Dataset) ---")
    base_resnet = models.resnet50(pretrained=False)
    client_true = ClientModel_True(base_resnet).to(device)
    server_true = ServerModel(base_resnet, num_priv_classes).to(device)

    optimizer_full = optim.Adam(list(client_true.parameters()) + list(server_true.parameters()), lr=0.001)

    for epoch in range(30):
        client_true.train()
        server_true.train()
        for inputs, targets in loader_priv_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_full.zero_grad()
            outputs = server_true(client_true(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_full.step()

    acc_target = evaluate(client_true, server_true, loader_priv_test, device)
    print(f"Target Model 在私有数据上的准确率: {acc_target:.2f}%")

    print("\n--- Phase 3, Step 1: MTIA - 迁移学习补全模型 ---")
    substitute_client = SubstituteClientModel().to(device)

    for param in server_true.parameters():
        param.requires_grad = False

    original_fc = server_true.fc
    fc_pub = nn.Linear(2048, num_pub_classes).to(device)
    server_true.fc = fc_pub

    optimizer_step1 = optim.Adam(list(substitute_client.parameters()) + list(fc_pub.parameters()), lr=0.001)

    for epoch in range(30):
        substitute_client.train()
        server_true.train()
        for inputs, targets in loader_pub_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_step1.zero_grad()
            features = substitute_client(inputs)
            outputs = server_true(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_step1.step()

    server_true.fc = original_fc
    acc_step1 = evaluate(substitute_client, server_true, loader_priv_test, device)
    print(f"仅完成 Step 1 (迁移学习) 后，近似模型准确率: {acc_step1:.2f}%")

    print("\n--- Phase 3, Step 2: MTIA - 自下而上自注意力对齐 (Bottom-up Layer-wise SAD) ---")
    server_true.fc = fc_pub
    alpha = 10.0

    print("  -> Round 1: 对齐替代块的深层 (layer2)，并保留分类损失")
    optimizer_step2_r1 = optim.Adam(substitute_client.parameters(), lr=0.0001)
    for epoch in range(15):
        substitute_client.train()
        server_true.eval()
        for inputs, targets in loader_pub_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_step2_r1.zero_grad()

            out1, out2 = substitute_client(inputs, return_all=True)
            outputs, server_feat = server_true(out2, return_feature=True)

            loss_cls = criterion(outputs, targets)
            loss_sad1 = sad_loss(out2, server_feat)
            loss = loss_cls + alpha * loss_sad1

            loss.backward()
            optimizer_step2_r1.step()

    for param in substitute_client.layer2.parameters():
        param.requires_grad = False

    print("  -> Round 2: 冻结 layer2，向前对齐更浅的层 (layer1)")
    optimizer_step2_r2 = optim.Adam(substitute_client.layer1.parameters(), lr=0.0001)
    for epoch in range(15):
        substitute_client.train()
        server_true.eval()
        for inputs, targets in loader_pub_train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_step2_r2.zero_grad()

            out1, out2 = substitute_client(inputs, return_all=True)
            outputs, _ = server_true(out2, return_feature=True)

            loss_cls = criterion(outputs, targets)
            loss_sad2 = sad_loss(out1, out2.detach())
            loss = loss_cls + alpha * loss_sad2

            loss.backward()
            optimizer_step2_r2.step()

    server_true.fc = original_fc
    acc_step2 = evaluate(substitute_client, server_true, loader_priv_test, device)
    print(f"完成 Step 2 逐层对齐后，最终近似模型准确率: {acc_step2:.2f}%")


if __name__ == "__main__":
    main()