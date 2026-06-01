import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import json
import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights

from models import ClientModel_True, ServerModel, SubstituteClientModel
from train import get_dataloaders, train_target_model, evaluate
from mtia import run_mtia_step1, run_mtia_step2

def load_config():
    parser = argparse.ArgumentParser(description="MTIA Attack Reproduction")
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--target_epochs', type=int, default=None)
    parser.add_argument('--step1_epochs', type=int, default=None)
    parser.add_argument('--step2_epochs', type=int, default=None)
    parser.add_argument('--alpha_sad', type=float, default=None)
    parser.add_argument('--force_retrain', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
            
    return config, args.force_retrain

def main():
    config, force_retrain = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['weights_dir'], exist_ok=True)
    weight_filename = f"target_ep{config['target_epochs']}_bs{config['batch_size']}_lr{config['target_lr']}.pth"
    weight_path = os.path.join(config['weights_dir'], weight_filename)

    # 1. 准备数据与模型 (启用 ImageNet 预训练加速收敛)
    loader_priv_train, loader_priv_test, loader_pub_train = get_dataloaders(config)
    base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    client_true = ClientModel_True(base_resnet).to(device)
    server_true = ServerModel(base_resnet, config['num_priv_classes']).to(device)

    # 2. Phase 1 & 2: 目标模型预训练与“协同推理”测试
    print("\n--- Phase 1 & 2: Target Model 预训练与协同推理评估 ---")
    if os.path.exists(weight_path) and not force_retrain:
        print(f"  检测到缓存，直接加载权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        client_true.load_state_dict(checkpoint['client'])
        server_true.load_state_dict(checkpoint['server'])
    else:
        print("  未检测到缓存，开始全量训练...")
        train_target_model(client_true, server_true, loader_priv_train, config, device)
        torch.save({'client': client_true.state_dict(), 'server': server_true.state_dict()}, weight_path)
        print(f"  权重已保存至: {weight_path}")

    # 此处即为 Phase 2: 模拟双端协同推理的基线准确率
    acc_collab = evaluate(client_true, server_true, loader_priv_test, device)
    print(f"\n[!] 正常协同推理 (C持有前段, S持有后段) 在私有数据上的准确度: {acc_collab:.2f}%")

    # 3. Phase 3, Step 1: 迁移学习
    print("\n--- Phase 3, Step 1: MTIA - 迁移学习补全模型 ---")
    substitute_client = SubstituteClientModel().to(device)
    original_fc = server_true.fc
    
    fc_pub = run_mtia_step1(substitute_client, server_true, loader_pub_train, original_fc, config, device)
    
    # 额外评估：测试替代模型在公有数据集上的准确度
    server_true.fc = fc_pub
    acc_pub_step1 = evaluate(substitute_client, server_true, loader_pub_train, device)
    print(f"[!] 攻击者在公有数据集(Pub)上的伪造准确度: {acc_pub_step1:.2f}%")
    
    server_true.fc = original_fc
    acc_step1 = evaluate(substitute_client, server_true, loader_priv_test, device)
    print(f"[!] 仅完成 Step 1 后，对私有数据(Priv)的窃取准确度: {acc_step1:.2f}%")

    # 4. Phase 3, Step 2: SAD 对齐
    print("\n--- Phase 3, Step 2: MTIA - 自注意力对齐 (SAD) ---")
    run_mtia_step2(substitute_client, server_true, loader_pub_train, original_fc, fc_pub, config, device)
    
    acc_step2 = evaluate(substitute_client, server_true, loader_priv_test, device)
    print(f"\n[!!!] 完成 Step 2 对齐后，对私有数据(Priv)的最终窃取准确度: {acc_step2:.2f}%")

if __name__ == "__main__":
    main()