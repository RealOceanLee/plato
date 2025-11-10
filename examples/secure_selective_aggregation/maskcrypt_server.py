# maskcrypt_server.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from config_manager import ConfigManager
from weight_processor import WeightProcessor
from ckks_encoder import CKKSEncoder


class MaskCryptServer:
    """MaskCrypt服务器，处理掩码共识和权重聚合"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.encoder = CKKSEncoder()
        self.processor = WeightProcessor(self.encoder)

        # 服务器状态
        self.client_info = {}
        self.consensus_mask = []
        self.global_weights = None
        self.current_round = 0

    def set_global_weights(self, global_weights: Dict[str, torch.Tensor]):
        """设置全局权重"""
        self.global_weights = global_weights

    def process_client_data(self, client_id: int, client_data: Dict[str, Any]):
        """处理客户端上传的数据"""
        if client_data['type'] == 'mask_proposal':
            # 处理掩码提案
            self.client_info[client_id] = {
                'mask_proposal': client_data['mask_proposal'],
                'priority': client_data['priority'],
                'data_size': client_data['data_size']
            }
        elif client_data['type'] == 'encrypted_weights':
            # 处理加密权重
            if client_id not in self.client_info:
                self.client_info[client_id] = {}

            self.client_info[client_id].update({
                'processed_weights': client_data['processed_weights'],
                'priority': client_data['priority'],
                'data_size': client_data['data_size']
            })

    def build_consensus_mask(self, total_parameters: int) -> List[int]:
        """构建共识掩码"""
        if not self.client_info:
            return []

        # 提取提案和优先级
        proposals = [info['mask_proposal'] for info in self.client_info.values()
                     if 'mask_proposal' in info]
        priorities = [info['priority'] for info in self.client_info.values()
                      if 'priority' in info]

        if not proposals:
            return []

        # 归一化优先级
        total_priority = sum(priorities)
        if total_priority == 0:
            normalized_priorities = [1.0 / len(priorities)] * len(priorities)
        else:
            normalized_priorities = [p / total_priority for p in priorities]

        # 创建权重投票系统
        param_votes = {}
        for proposal, weight in zip(proposals, normalized_priorities):
            for param_idx in proposal:
                if param_idx in param_votes:
                    param_votes[param_idx] += weight
                else:
                    param_votes[param_idx] = weight

        # 按投票权重排序并选择前total_mask_size个
        total_mask_size = int(self.config.encrypt_ratio * total_parameters)
        sorted_params = sorted(param_votes.items(), key=lambda x: x[1], reverse=True)
        self.consensus_mask = [param_idx for param_idx, _ in sorted_params[:total_mask_size]]

        return self.consensus_mask

    def aggregate_weights(self) -> Dict[str, torch.Tensor]:
        """聚合权重：基于优先级的加权平均"""
        if not self.client_info or self.global_weights is None:
            return self.global_weights

        # 收集客户端权重和优先级
        client_weights = []
        client_priorities = []

        for client_id, info in self.client_info.items():
            if 'processed_weights' in info:
                # 重建客户端权重
                weights = self.processor.reconstruct_weights(info['processed_weights'])
                client_weights.append(weights)
                client_priorities.append(info['priority'])

        if not client_weights:
            return self.global_weights

        # 归一化优先级
        total_priority = sum(client_priorities)
        if total_priority == 0:
            weights = [1.0 / len(client_priorities)] * len(client_priorities)
        else:
            weights = [p / total_priority for p in client_priorities]

        # 初始化聚合权重
        aggregated_weights = {}
        for key in self.global_weights.keys():
            aggregated_weights[key] = torch.zeros_like(self.global_weights[key])

        # 加权聚合
        for client_weight, weight in zip(client_weights, weights):
            for key in client_weight.keys():
                if key in aggregated_weights:
                    aggregated_weights[key] += weight * client_weight[key]

        return aggregated_weights

    def get_consensus_mask(self) -> List[int]:
        """获取共识掩码"""
        return self.consensus_mask

    def clear_client_info(self):
        """清空客户端信息"""
        self.client_info.clear()

    def increment_round(self):
        """增加轮次"""
        self.current_round += 1