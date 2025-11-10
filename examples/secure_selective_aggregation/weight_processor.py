# weight_processor.py
import numpy as np
import torch
from typing import Dict, Any, Tuple, List
from ckks_encoder import CKKSEncoder, VectorOperations


class WeightProcessor:
    """权重处理器，处理完整权重矩阵的上传和下载"""

    def __init__(self, encoder: CKKSEncoder):
        self.encoder = encoder
        self.vector_ops = VectorOperations(encoder)

    def prepare_upload_weights(self, local_weights: Dict[str, torch.Tensor],
                               global_weights: Dict[str, torch.Tensor],
                               mask_indices: List[int]) -> Dict[str, Any]:
        """
        准备上传的权重：完整权重矩阵 + 加密部分的多项式编码
        """
        # 展平权重
        local_flat = self._flatten_weights(local_weights)
        global_flat = self._flatten_weights(global_weights)

        # 对于不需要加密的轮次，直接使用完整本地权重
        if mask_indices is None or len(mask_indices) == 0:
            return {
                'full_weights': local_flat,
                'encoded_pm1': None,
                'encoded_pm2': None,
                'mask_indices': np.array([], dtype=int),
                'original_shape': {k: v.shape for k, v in local_weights.items()}
            }

        # 对于需要加密的轮次，使用全局权重替换非掩码位置
        mask_indices_np = np.array(mask_indices)

        # 创建混合权重：掩码位置使用本地权重，非掩码位置使用全局权重
        mixed_weights = global_flat.copy()
        mixed_weights[mask_indices_np] = local_flat[mask_indices_np]

        # 编码掩码部分
        full_weights, encoded_pm1, encoded_pm2 = self.encoder.encode_weights(
            mixed_weights, mask_indices_np
        )

        return {
            'full_weights': full_weights,
            'encoded_pm1': encoded_pm1,
            'encoded_pm2': encoded_pm2,
            'mask_indices': mask_indices_np,
            'original_shape': {k: v.shape for k, v in local_weights.items()}
        }

    def reconstruct_weights(self, processed_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """从处理后的数据重建权重字典"""
        full_weights = processed_data['full_weights']
        encoded_pm1 = processed_data['encoded_pm1']
        mask_indices = processed_data['mask_indices']
        original_shape = processed_data['original_shape']

        # 如果有密文部分，解码并合并
        if encoded_pm1 is not None:
            full_weights = self.encoder.decode_weights(full_weights, encoded_pm1, mask_indices)

        # 重建权重字典
        return self._unflatten_weights(full_weights, original_shape)

    def compute_similarity(self, weights_a: Dict[str, Any], weights_b: Dict[str, Any]) -> float:
        """计算两个权重集的相似度"""
        full_a = weights_a['full_weights']
        full_b = weights_b['full_weights']

        # 计算余弦相似度
        dot_product = self.vector_ops.compute_inner_product(full_a, full_b)
        norm_a = self.vector_ops.compute_l2_norm(full_a)
        norm_b = self.vector_ops.compute_l2_norm(full_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def compute_correlation(self, weights_a: Dict[str, Any], weights_b: Dict[str, Any]) -> float:
        """计算两个权重集的皮尔逊相关系数"""
        full_a = weights_a['full_weights']
        full_b = weights_b['full_weights']
        return self.vector_ops.compute_pearson_correlation(full_a, full_b)

    def _flatten_weights(self, weights_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """展平权重字典"""
        flattened = []
        for tensor in weights_dict.values():
            flattened.extend(tensor.detach().cpu().flatten().numpy())
        return np.array(flattened)

    def _unflatten_weights(self, flattened_vector: np.ndarray,
                           original_shape: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """从展平向量重建权重字典"""
        weights_dict = {}
        start_idx = 0

        for key, shape in original_shape.items():
            size = np.prod(shape)
            segment = flattened_vector[start_idx:start_idx + size]
            weights_dict[key] = torch.tensor(segment.reshape(shape))
            start_idx += size

        return weights_dict