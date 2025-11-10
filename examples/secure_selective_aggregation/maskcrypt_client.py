# maskcrypt_client.py
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from config_manager import ConfigManager
from ckks_encoder import CKKSEncoder, VectorOperations
from weight_processor import WeightProcessor


class MaskCryptClient:
    """MaskCrypt客户端，支持选择性加密和完整权重上传"""

    def __init__(self, client_id: int, config: ConfigManager):
        self.client_id = client_id
        self.config = config
        self.encoder = CKKSEncoder(N=8192, scaling_factor=2 ** 40)
        self.vector_ops = VectorOperations(self.encoder)
        self.processor = WeightProcessor(self.encoder)

        # 客户端状态
        self.data_size = 0
        self.correlation = 0.0
        self.priority = 0.0
        self.mask_proposal = []
        self.local_weights = None
        self.global_weights = None

    def set_data_size(self, data_size: int):
        """设置客户端数据量"""
        self.data_size = data_size

    def compute_mask_proposal(self, local_weights: Dict[str, torch.Tensor],
                              global_weights: Dict[str, torch.Tensor],
                              gradients: Dict[str, torch.Tensor]) -> Tuple[List[int], float]:
        """
        计算掩码提案，选择最重要的权重进行加密
        """
        self.local_weights = local_weights
        self.global_weights = global_weights

        # 展平权重和梯度
        local_flat = self._flatten_weights(local_weights)
        global_flat = self._flatten_weights(global_weights)
        gradients_flat = self._flatten_gradients(gradients)

        # 计算重要性分数
        importance = self.vector_ops.compute_weight_importance(local_flat, gradients_flat)

        # 选择最重要的参数进行加密
        encrypt_ratio = self.config.encrypt_ratio
        mask_size = int(encrypt_ratio * len(local_flat))
        self.mask_proposal = np.argsort(importance)[-mask_size:].tolist()

        # 计算相关系数和优先级
        self.correlation = self.vector_ops.compute_pearson_correlation(local_flat, global_flat)

        # 估算总数据量（在实际应用中应从服务器获取）
        total_data_size = self.config.total_clients * self.data_size
        self.priority = self.vector_ops.compute_client_priority(
            self.data_size, total_data_size, self.correlation
        )

        return self.mask_proposal, self.priority

    def prepare_upload_data(self, current_round: int,
                            consensus_mask: List[int] = None) -> Dict[str, Any]:
        """
        准备上传数据：根据轮次决定上传掩码提案还是加密权重
        """
        if current_round % 2 != 0:
            # 奇数轮：上传掩码提案
            return {
                'type': 'mask_proposal',
                'mask_proposal': self.mask_proposal,
                'priority': self.priority,
                'data_size': self.data_size
            }
        else:
            # 偶数轮：上传加密权重
            if consensus_mask is None:
                consensus_mask = []

            processed_weights = self.processor.prepare_upload_weights(
                self.local_weights, self.global_weights, consensus_mask
            )

            return {
                'type': 'encrypted_weights',
                'processed_weights': processed_weights,
                'priority': self.priority,
                'data_size': self.data_size
            }

    def update_global_weights(self, global_weights: Dict[str, torch.Tensor]):
        """更新全局权重"""
        self.global_weights = global_weights

    def _flatten_weights(self, weights_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """展平权重字典"""
        flattened = []
        for tensor in weights_dict.values():
            flattened.extend(tensor.detach().cpu().flatten().numpy())
        return np.array(flattened)

    def _flatten_gradients(self, gradients_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """展平梯度字典"""
        flattened = []
        for tensor in gradients_dict.values():
            flattened.extend(tensor.detach().cpu().flatten().numpy())
        return np.array(flattened)

    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            'client_id': self.client_id,
            'data_size': self.data_size,
            'correlation': self.correlation,
            'priority': self.priority,
            'mask_proposal': self.mask_proposal
        }