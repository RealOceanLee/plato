# maskcrypt_client.py
import torch
import copy
from torch.utils.data import DataLoader, Subset


class MaskCryptClient:
    def __init__(self, client_id, config, dataset, is_byzantine=False, model_class=None):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.is_byzantine = is_byzantine
        self.model_class = model_class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 攻击类型从配置读取，默认为 random_weights
        self.attack_type = config.get('byzantine_attack_type', 'random')

        # 缓存本地结果
        self.local_weights = None
        self.weight_update = None
        self.global_weights = None

        # 初始化时打印拜占庭身份
        if self.is_byzantine:
            print(f"  😈 客户端 {self.client_id} 初始化为拜占庭节点 | 攻击类型: {self.attack_type}")

    def get_data(self, max_samples=200):
        """用于预聚类阶段采样少量明文数据"""
        if len(self.dataset) == 0:
            raise ValueError(f"客户端 {self.client_id} 数据集为空")
        n_samples = min(max_samples, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:n_samples]
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=n_samples, shuffle=False)
        data, targets = next(iter(loader))
        return data.to(self.device), targets.to(self.device)

    def update_global_weights(self, global_weights):
        # ✅ 优化：直接存储权重，不改变设备
        self.global_weights = copy.deepcopy(global_weights)

    def local_train(self, global_weights, epochs=1, current_round=1):
        # ✅ 优化：直接使用传入的权重，避免设备转换
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.model_class().to(self.device)

        # 直接加载权重到设备
        self.model.load_state_dict(global_weights)

        self.model.train()

        # ✅ 安全获取学习率
        lr = self.config.get('learning_rate', 0.01)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # ✅ 优化：简化数据加载器配置
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # ✅ 获取本地权重
        local_weights = {k: v for k, v in self.model.state_dict().items()}
        weight_update = {k: local_weights[k] - global_weights[k] for k in global_weights}

        # ✅ 拜占庭攻击注入
        if self.is_byzantine:
            if self.attack_type == "random" or self.attack_type == "random_weights":
                # 随机权重攻击
                for k in local_weights:
                    local_weights[k] = torch.randn_like(local_weights[k])
                weight_update = {k: local_weights[k] - global_weights[k] for k in global_weights}
                print(f"  😈 [Round {current_round}] 客户端 {self.client_id} 发起攻击: random_weights")

            elif self.attack_type == "sign_flip":
                # 符号翻转攻击
                weight_update = {k: -v for k, v in weight_update.items()}
                local_weights = {k: global_weights[k] + weight_update[k] for k in global_weights}
                print(f"  😈 [Round {current_round}] 客户端 {self.client_id} 发起攻击: sign_flip")

            elif self.attack_type == "zero_update":
                # 零更新攻击
                weight_update = {k: torch.zeros_like(v) for k, v in weight_update.items()}
                local_weights = copy.deepcopy(global_weights)
                print(f"  😈 [Round {current_round}] 客户端 {self.client_id} 发起攻击: zero_update")

            elif self.attack_type == "scaled_update":
                # 缩放更新攻击
                scale = self.config.get('byzantine_attack_scale', 10.0)
                weight_update = {k: scale * v for k, v in weight_update.items()}
                local_weights = {k: global_weights[k] + weight_update[k] for k in global_weights}
                print(f"  😈 [Round {current_round}] 客户端 {self.client_id} 发起攻击: scaled_update (×{scale})")

            else:
                # 未知攻击类型，回退到随机权重
                for k in local_weights:
                    local_weights[k] = torch.randn_like(local_weights[k])
                weight_update = {k: local_weights[k] - global_weights[k] for k in global_weights}
                print(f"  ⚠️ [Round {current_round}] 客户端 {self.client_id} 使用未知攻击 '{self.attack_type}' → 回退到 random_weights")

        self.local_weights = local_weights
        self.weight_update = weight_update

        return local_weights, 0.0

    def prepare_upload_data_simple(self, round_num):
        return {
            'local_weights': self.local_weights,
            'weight_update': self.weight_update
        }