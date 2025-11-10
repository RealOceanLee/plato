# ckks_encoder.py
import numpy as np
import torch
from typing import Tuple, Optional


class CKKSEncoder:
    """CKKS编码器，支持部分权重明文传输和多项式编码"""

    def __init__(self, N: int = 8192, scaling_factor: int = 2 ** 40):
        self.N = N
        self.scaling_factor = scaling_factor

    def pm1(self, vector: np.ndarray) -> np.ndarray:
        """第一种打包方式：Σ m_i * X^i"""
        if len(vector) > self.N:
            raise ValueError(f"Vector length {len(vector)} exceeds polynomial degree {self.N}")

        padded_vector = np.zeros(self.N)
        padded_vector[:len(vector)] = vector

        polynomial = np.fft.fft(padded_vector)
        encoded_poly = np.round(polynomial * self.scaling_factor).astype(np.int64)

        return encoded_poly

    def pm2(self, vector: np.ndarray) -> np.ndarray:
        """第二种打包方式：-Σ m_i * X^{N-i}"""
        if len(vector) > self.N:
            raise ValueError(f"Vector length {len(vector)} exceeds polynomial degree {self.N}")

        padded_vector = np.zeros(self.N)
        padded_vector[:len(vector)] = vector

        reversed_vector = -padded_vector[::-1]
        polynomial = np.fft.fft(reversed_vector)
        encoded_poly = np.round(polynomial * self.scaling_factor).astype(np.int64)

        return encoded_poly

    def encode_weights(self, weights: np.ndarray, mask_indices: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        编码权重：部分明文，部分加密
        返回完整权重矩阵和两个多项式编码
        """
        # 创建完整权重矩阵（所有位置都有值）
        full_weights = weights.copy()

        # 创建仅包含掩码位置的向量用于加密
        mask_weights = np.zeros_like(weights)
        mask_weights[mask_indices] = weights[mask_indices]

        # 对掩码部分进行多项式编码
        encoded_pm1 = self.pm1(mask_weights)
        encoded_pm2 = self.pm2(mask_weights)

        return full_weights, encoded_pm1, encoded_pm2

    def decode_weights(self, full_weights: np.ndarray, encoded_pm1: np.ndarray,
                       mask_indices: np.ndarray) -> np.ndarray:
        """解码权重：合并明文和密文部分"""
        if encoded_pm1 is not None:
            # 解码密文部分
            decoded_mask = self.decode(encoded_pm1)
            # 用解码的值替换掩码位置
            full_weights[mask_indices] = decoded_mask[mask_indices]

        return full_weights

    def decode(self, polynomial: np.ndarray) -> np.ndarray:
        """解码多项式为向量"""
        coefficients = np.fft.ifft(polynomial / self.scaling_factor)
        vector = np.real(coefficients)
        return vector


class VectorOperations:
    """向量操作，支持内积和相关系数计算"""

    def __init__(self, encoder: CKKSEncoder):
        self.encoder = encoder

    def compute_inner_product(self, weights_a: np.ndarray, weights_b: np.ndarray) -> float:
        """计算两个权重向量的内积"""
        return np.dot(weights_a, weights_b)

    def compute_l2_norm(self, weights: np.ndarray) -> float:
        """计算权重向量的L2范数"""
        return np.linalg.norm(weights)

    def compute_pearson_correlation(self, weights_a: np.ndarray, weights_b: np.ndarray) -> float:
        """计算皮尔逊相关系数"""
        if len(weights_a) != len(weights_b):
            raise ValueError("Vectors must have the same length")

        # 计算均值
        mean_a = np.mean(weights_a)
        mean_b = np.mean(weights_b)

        # 中心化
        centered_a = weights_a - mean_a
        centered_b = weights_b - mean_b

        # 计算协方差和标准差
        covariance = np.dot(centered_a, centered_b)
        std_a = np.std(centered_a)
        std_b = np.std(centered_b)

        if std_a == 0 or std_b == 0:
            return 0.0

        correlation = covariance / (std_a * std_b)
        return np.clip(correlation, -1.0, 1.0)

    def compute_weight_importance(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """基于梯度和权重计算重要性分数"""
        return np.abs(weights * gradients)

    def compute_client_priority(self, data_size: int, total_data_size: int,
                                correlation: float, tau: float = 0.5) -> float:
        """
        计算客户端优先级：nk/n + τl
        nk: 客户端数据量
        n: 总数据量
        τl: 相关系数
        """
        data_ratio = data_size / total_data_size if total_data_size > 0 else 0
        priority = data_ratio + tau * correlation
        return priority