import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_attention_map(A, temperature=10.0):
    """
    计算特征图的空间注意力分布
    引入 temperature (温度系数) 以防止激活值平方和过大导致 Softmax 梯度饱和死亡
    """
    # 计算通道维度的绝对值平方和 (B, 1, H, W)
    G = torch.sum(torch.abs(A) ** 2, dim=1, keepdim=True)
    B, _, H, W = G.size()
    G_flat = G.view(B, -1)
    
    # 缩放后进行 Softmax，盘活梯度
    Phi_flat = F.softmax(G_flat / temperature, dim=1)
    return Phi_flat.view(B, 1, H, W)

def sad_loss(A_shallow, A_deep, temperature=10.0):
    """
    计算自注意力对齐损失，修复了 Batch 累加问题，改为按样本求 L2 均值
    """
    phi_shallow = get_attention_map(A_shallow, temperature)
    phi_deep = get_attention_map(A_deep, temperature)
    
    # 将深层注意力图上采样到浅层大小
    phi_deep_up = F.interpolate(phi_deep, size=phi_shallow.shape[2:], mode='bilinear', align_corners=False)
    
    # 【修复】：计算每个样本独立的 L2 范数，然后再对 Batch 求平均
    diff = (phi_shallow - phi_deep_up).view(phi_shallow.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean()

def run_mtia_step1(sub_client, server, loader_pub, original_fc, config, device):
    # 严格冻结服务端原有的深层特征提取器
    for param in server.parameters():
        param.requires_grad = False
        
    # 替换公共数据集分类头
    fc_pub = nn.Linear(server.fc.in_features, config['num_pub_classes']).to(device)
    server.fc = fc_pub
    
    optimizer = optim.Adam(list(sub_client.parameters()) + list(server.fc.parameters()), lr=config['step1_lr'])
    criterion = nn.CrossEntropyLoss()
    
    print("  => 开始 Step 1 (迁移学习) 训练...")
    for epoch in range(config['step1_epochs']):
        sub_client.train()
        server.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader_pub:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = server(sub_client(inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_acc = 100. * correct / total
        print(f"     Epoch [{epoch+1}/{config['step1_epochs']}] Pub Data Loss: {running_loss/len(loader_pub):.4f} | Pub Acc: {epoch_acc:.2f}%")
            
    server.fc = original_fc
    return fc_pub

def run_mtia_step2(sub_client, server, loader_pub, original_fc, fc_pub, config, device):
    server.fc = fc_pub 
    criterion = nn.CrossEntropyLoss()
    alpha = config['alpha_sad']
    
    # 推荐的温度系数，可根据实际特征图尺度调整
    T = 5.0 

    print("  => [Round 1] 对齐替代模型深层 (冻结 Server)...")
    optimizer_r1 = optim.Adam(sub_client.parameters(), lr=config['step2_lr'])
    for epoch in range(config['step2_epochs']):
        sub_client.train()
        server.eval() 
        running_loss_cls = 0.0
        running_loss_sad = 0.0
        
        for inputs, targets in loader_pub:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_r1.zero_grad()
            out1, out2 = sub_client(inputs, return_all=True)
            outputs, server_feat = server(out2, return_feature=True)
            
            loss_cls = criterion(outputs, targets)
            loss_sad_val = sad_loss(out2, server_feat.detach(), temperature=T) 
            
            loss = loss_cls + alpha * loss_sad_val
            loss.backward()
            optimizer_r1.step()
            
            running_loss_cls += loss_cls.item()
            running_loss_sad += loss_sad_val.item()
            
        print(f"     Epoch [{epoch+1}/{config['step2_epochs']}] Cls Loss: {running_loss_cls/len(loader_pub):.4f} | SAD Loss: {running_loss_sad/len(loader_pub):.4f}")

    # 冻结已经对齐的 VGG 深层
    for param in sub_client.layer2.parameters():
        param.requires_grad = False

    print("  => [Round 2] 对齐替代模型浅层 (冻结 layer2)...")
    optimizer_r2 = optim.Adam(sub_client.layer1.parameters(), lr=config['step2_lr'])
    for epoch in range(config['step2_epochs']):
        sub_client.train()
        server.eval()
        running_loss_cls = 0.0
        running_loss_sad = 0.0
        
        for inputs, targets in loader_pub:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_r2.zero_grad()
            out1, out2 = sub_client(inputs, return_all=True)
            outputs, _ = server(out2, return_feature=True)
            
            loss_cls = criterion(outputs, targets)
            loss_sad_val = sad_loss(out1, out2.detach(), temperature=T)
            
            loss = loss_cls + alpha * loss_sad_val
            loss.backward()
            optimizer_r2.step()
            
            running_loss_cls += loss_cls.item()
            running_loss_sad += loss_sad_val.item()
            
        print(f"     Epoch [{epoch+1}/{config['step2_epochs']}] Cls Loss: {running_loss_cls/len(loader_pub):.4f} | SAD Loss: {running_loss_sad/len(loader_pub):.4f}")

    server.fc = original_fc