#!/usr/bin/env python3
import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from maskcrypt_client import MaskCryptClient
from maskcrypt_server import MaskCryptServer
from data_loader import load_mnist_data, create_non_iid_data
from lenet5 import LeNet5


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    config.total_clients = 20

    train_dataset, test_dataset = load_mnist_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    client_datasets = create_non_iid_data(train_dataset, num_clients=config.total_clients)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = LeNet5().to(device)

    all_clients = []
    byzantine_client_ids = config.get('byzantine_clients', [3, 7, 12, 18])

    for client_id in range(config.total_clients):
        is_byzantine = client_id in byzantine_client_ids
        client = MaskCryptClient(
            client_id,
            config,
            client_datasets[client_id],
            is_byzantine=is_byzantine,
            model_class=LeNet5
        )
        all_clients.append(client)

    # ✅ 打印拜占庭信息
    print(f"\n{'=' * 60}")
    print(f"✅ 联邦学习配置")
    print(f"   总客户端数: {config.total_clients}")
    print(f"   拜占庭节点: {sorted(byzantine_client_ids)}")
    print(f"   攻击类型: {config.get('byzantine_attack_type', 'random')}")
    print(f"   防御机制: Krum (f=1)")
    print(f"   预聚类方法: SnapCFL")
    print(f"   设备: {device} {'(' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"{'=' * 60}\n")

    server = MaskCryptServer(config)
    server.update_clusters_with_snapcfl(all_clients)

    # ✅ 优化：直接设置GPU权重，避免CPU转换
    server.set_global_weights({k: v for k, v in global_model.state_dict().items()})

    best_accuracy = 0.0
    for round_num in range(1, config.rounds + 1):
        print(f"\n=== Round {round_num} ===")

        # ✅ 优化：直接传递GPU权重，避免CPU转换
        global_weights_gpu = {k: v for k, v in global_model.state_dict().items()}
        for client in all_clients:
            client.update_global_weights(global_weights_gpu)

        for client in all_clients:
            local_weights, loss = client.local_train(
                global_weights_gpu,  # ✅ 直接传递GPU权重
                epochs=config.get('epochs', config.get('local_epochs', 1)),
                current_round=round_num
            )
            if local_weights is not None:
                upload_data = client.prepare_upload_data_simple(round_num)
                server.process_client_data(client.client_id, upload_data)

        aggregated_weights = server.aggregate_with_clustered_krum(f=1)
        if aggregated_weights:
            # ✅ 优化：聚合后的权重直接加载到GPU模型
            global_model.load_state_dict(aggregated_weights)
            server.set_global_weights({k: v for k, v in global_model.state_dict().items()})
            accuracy = evaluate_model(global_model, test_loader, device)
            print(f"📊 全局准确率: {accuracy:.2f}%")
            best_accuracy = max(best_accuracy, accuracy)

        server.clear_client_info()
        server.increment_round()

    print(f"\n🎉 训练完成！最佳准确率: {best_accuracy:.2f}%")
    print(f"❗ 拜占庭节点被 Krum 选中的次数: {server.byzantine_selection_count}")


if __name__ == "__main__":
    main()