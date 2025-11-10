# config_manager.py
import toml
from typing import Dict, Any


class ConfigManager:
    """配置管理器，处理TOML配置文件"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载TOML配置文件"""
        with open(self.config_path, 'r') as f:
            return toml.load(f)

    @property
    def clients(self) -> Dict[str, Any]:
        """获取客户端配置"""
        return self.config.get('clients', {})

    @property
    def server(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self.config.get('server', {})

    @property
    def data(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})

    @property
    def trainer(self) -> Dict[str, Any]:
        """获取训练器配置"""
        return self.config.get('trainer', {})

    @property
    def algorithm(self) -> Dict[str, Any]:
        """获取算法配置"""
        return self.config.get('algorithm', {})

    @property
    def encrypt_ratio(self) -> float:
        """获取加密比例"""
        return self.clients.get('encrypt_ratio', 0.05)

    @property
    def random_mask(self) -> bool:
        """获取是否使用随机掩码"""
        return self.clients.get('random_mask', False)

    @property
    def total_clients(self) -> int:
        """获取客户端总数"""
        return self.clients.get('total_clients', 10)

    @property
    def per_round(self) -> int:
        """获取每轮选择的客户端数"""
        return self.clients.get('per_round', 2)

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self.trainer.get('model_name', 'lenet5')

    @property
    def rounds(self) -> int:
        """获取训练轮数"""
        return self.trainer.get('rounds', 50)

    @property
    def epochs(self) -> int:
        """获取本地训练轮数"""
        return self.trainer.get('epochs', 2)

    @property
    def batch_size(self) -> int:
        """获取批次大小"""
        return self.trainer.get('batch_size', 32)