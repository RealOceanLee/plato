# config_manager.py
class ConfigManager:
    """配置管理器 - 支持扁平化 TOML 加载"""
    def __init__(self, config_file=None):
        # 基础配置
        self.total_clients = 10
        self.clusters = 3
        self.total_rounds = 50
        self.local_epochs = 1
        self.batch_size = 32
        self.learning_rate = 0.01
        self.random_seed = 1

        # 拜占庭配置（关键）
        self.byzantine_clients = 0  # 可为整数（数量）或列表（ID）
        self.byzantine_attack_type = "random"  # 可为"random"或具体攻击类型
        self.byzantine_start_round = 5
        self.byzantine_threshold = 0.3
        self.byzantine_attack_scale = 10.0  # scaled_update 的攻击缩放因子

        # L2 正则化
        self.weight_decay = 1e-4
        self.adaptive_weight_decay = True

        # 其他
        self.per_round = 10
        self.datasource = "Torchvision"
        self.dataset_name = "MNIST"
        self.download = True
        self.partition_size = 1000
        self.sampler = "iid"
        self.model_name = "lenet5"
        self.target_accuracy = 0.99
        self.max_concurrency = 1
        self.do_test = False

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        try:
            import toml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = toml.load(f)
            flat_config = self._flatten_dict(config_data)
            self.update_from_dict(flat_config)
            print(f"Configuration loaded from {config_file}")
        except ImportError:
            print("toml module not available, using default configuration")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def update_from_dict(self, config_dict: dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Ignoring unknown config key: {key}")

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

    def get_adaptive_weight_decay(self, round_num: int, similarity: float) -> float:
        if not self.adaptive_weight_decay:
            return self.weight_decay
        base_decay = self.weight_decay
        similarity_factor = max(0, 1.0 - similarity)
        round_factor = min(1.0, round_num / self.total_rounds)
        adaptive_decay = base_decay * (1.0 + 2.0 * similarity_factor + round_factor)
        return adaptive_decay

    @property
    def rounds(self):
        return self.total_rounds

    @property
    def epochs(self):
        return self.local_epochs