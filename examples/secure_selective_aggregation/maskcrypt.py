#!/usr/bin/env python3
# maskcrypt.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from maskcrypt_client import MaskCryptClient
from maskcrypt_server import MaskCryptServer


class LeNet5(nn.Module):
    """LeNet-5模型"""

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MaskCrypt Federated Learning')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to configuration file')
    args = parser.parse_args()

    # 加载配置
    config = ConfigManager(args.config)
    print(f"Loaded configuration from {args.config}")

    # 初始化服务器和客户端
    server = MaskCryptServer(config)

    # 创建全局模型
    global_model = LeNet5()
    total_parameters = count_parameters(global_model)
    print(f"Model: {config.model_name}, Parameters: {total_parameters}")

    # 设置服务器全局权重
    server.set_global_weights(dict(global_model.named_parameters()))

    # 模拟客户端
    clients = []
    for i in range(config.per_round):
        client = MaskCryptClient(i, config)
        # 模拟数据量
        data_size = 1000  # 从配置或数据集中获取实际值
        client.set_data_size(data_size)
        clients.append(client)

    # 模拟训练过程
    for round_num in range(1, config.rounds + 1):
        print(f"\n=== Round {round_num} ===")
        server.current_round = round_num

        if round_num % 2 != 0:
            # 奇数轮：客户端计算并提交掩码提案
            print("Phase 1: Mask Proposal")
            for client in clients:
                # 模拟本地训练（实际中应该使用真实数据）
                local_model = LeNet5()
                gradients = {name: torch.randn_like(param)
                             for name, param in local_model.named_parameters()}

                # 计算掩码提案
                mask_proposal, priority = client.compute_mask_proposal(
                    dict(local_model.named_parameters()),
                    dict(global_model.named_parameters()),
                    gradients
                )

                # 准备上传数据
                upload_data = client.prepare_upload_data(round_num)
                server.process_client_data(client.client_id, upload_data)

                print(f"Client {client.client_id}: Priority={priority:.4f}, "
                      f"Mask size={len(mask_proposal)}")

            # 服务器构建共识掩码
            consensus_mask = server.build_consensus_mask(total_parameters)
            print(f"Consensus mask size: {len(consensus_mask)}")

        else:
            # 偶数轮：客户端使用共识掩码上传加密权重
            print("Phase 2: Encrypted Weights Upload")
            consensus_mask = server.get_consensus_mask()

            for client in clients:
                # 模拟本地训练后的模型
                local_model = LeNet5()
                client.local_weights = dict(local_model.named_parameters())
                client.global_weights = dict(global_model.named_parameters())

                # 准备上传数据
                upload_data = client.prepare_upload_data(round_num, consensus_mask)
                server.process_client_data(client.client_id, upload_data)

                print(f"Client {client.client_id}: Uploaded encrypted weights")

            # 服务器聚合权重
            aggregated_weights = server.aggregate_weights()
            global_model.load_state_dict(aggregated_weights)
            server.set_global_weights(dict(global_model.named_parameters()))

            print("Weights aggregated successfully")

            # 清空客户端信息，准备下一轮
            server.clear_client_info()

    print("\n=== Training Completed ===")
    print(f"Total rounds: {config.rounds}")
    print("MaskCrypt protocol executed successfully")


if __name__ == "__main__":
    main()