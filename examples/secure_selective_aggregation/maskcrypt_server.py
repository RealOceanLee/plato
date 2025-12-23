# maskcrypt_server.py
import torch
import numpy as np
from typing import Dict, Any, List
from config_manager import ConfigManager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)


class MaskCryptServer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client_info = {}
        self.client_clusters = {}  # 由 SnapCFL 预聚类生成
        self.global_weights = None
        self.current_round = 0
        self.byzantine_set = set(config.get('byzantine_clients', []))
        self.byzantine_selection_count = 0

    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        self.global_weights = weights

    def process_client_data(self, client_id: int, data: Dict[str, Any]):
        if data is None:
            return
        self.client_info[client_id] = {
            'local_weights': data.get('local_weights'),
            'weight_update': data.get('weight_update'),
            'cluster_id': self.client_clusters.get(client_id, 0)
        }

    def clear_client_info(self):
        self.client_info.clear()

    def increment_round(self):
        self.current_round += 1

    def _simple_average(self, updates_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg_delta = {}
        for key in updates_list[0].keys():
            avg_delta[key] = torch.stack([upd[key] for upd in updates_list]).mean(dim=0)
        return avg_delta

    def _compute_similarity_matrix(self, all_clients: List[Any]):
        """
        实现 SnapCFL 论文中的预聚类核心：
        对每一对客户端 (i, j)，训练二分类器判断其数据是否来自同一分布。
        分类准确率越高，说明分布越不同。
        """
        n = len(all_clients)
        similarity_matrix = np.full((n, n), 0.5)  # 默认随机水平
        client_ids = [c.client_id for c in all_clients]

        print(f"  🔍 构建 {n}x{n} 相似性矩阵（共 {n * (n - 1) // 2} 对）...")

        for i in range(n):
            for j in range(i + 1, n):
                client_i = all_clients[i]
                client_j = all_clients[j]

                try:
                    # 获取少量明文数据（仅用于预聚类）
                    X_i, y_i = client_i.get_data(max_samples=200)
                    X_j, y_j = client_j.get_data(max_samples=200)

                    # 合并数据并打伪标签：client_i → 0, client_j → 1
                    X_combined = torch.cat([X_i, X_j], dim=0)
                    X_flat = X_combined.view(X_combined.size(0), -1).cpu().numpy()
                    y_pseudo = np.array([0] * X_i.size(0) + [1] * X_j.size(0))

                    # 划分训练/测试集
                    if len(np.unique(y_pseudo)) < 2:
                        acc = 0.5
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_flat, y_pseudo, test_size=0.3,
                            stratify=y_pseudo, random_state=42
                        )

                        # 标准化 + 训练轻量级分类器
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        clf = LogisticRegression(max_iter=1000, random_state=42)
                        clf.fit(X_train_scaled, y_train)
                        acc = clf.score(X_test_scaled, y_test)

                    # 存储准确率（作为 dissimilarity 度量）
                    similarity_matrix[i][j] = acc
                    similarity_matrix[j][i] = acc

                except Exception as e:
                    print(f"    ⚠️ 客户端 {client_i.client_id}-{client_j.client_id} 失败: {e}")
                    # 保持默认 0.5

        return similarity_matrix, client_ids

    def update_clusters_with_snapcfl(self, all_clients: List[Any]):
        """
        执行 SnapCFL 预聚类（仅调用一次）
        """
        print("\n🔍 [SnapCFL 预聚类] 基于数据分布相似性分组...")
        sim_matrix, client_ids = self._compute_similarity_matrix(all_clients)

        # 转换为距离矩阵：distance = |acc - 0.5|
        distance_matrix = np.abs(sim_matrix - 0.5)

        # 使用 DBSCAN 聚类（eps 可根据数据调整）
        clustering = DBSCAN(eps=0.15, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        # 映射回 client_id
        for idx, cid in enumerate(client_ids):
            self.client_clusters[cid] = int(cluster_labels[idx])

        # 统计结果
        unique_labels = set(cluster_labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = list(cluster_labels).count(-1)

        print(f"  ✅ 聚类完成：{num_clusters} 个簇，{noise_points} 个噪声点")
        for label in sorted(unique_labels):
            count = list(cluster_labels).count(label)
            members = [cid for idx, cid in enumerate(client_ids) if cluster_labels[idx] == label]
            print(f"    Cluster {label}: {count} clients → {sorted(members)}")

    def _krum_select(self, updates: List[Dict[str, torch.Tensor]], client_ids: List[int], f: int = 1) -> Dict[
        str, torch.Tensor]:
        """
        在给定的更新列表中使用Krum算法选择一个最可信的更新
        """
        n = len(updates)
        if n == 0:
            return None
        if n == 1:
            return updates[0]

        if n >= 2 * f + 2:
            # Krum 选择
            flat_updates = []
            for upd in updates:
                vec = torch.cat([v.flatten().float() for v in upd.values()])
                flat_updates.append(vec)
            flat_updates = torch.stack(flat_updates)

            diff = flat_updates.unsqueeze(1) - flat_updates.unsqueeze(0)
            distances = torch.sum(diff ** 2, dim=2)

            scores = []
            for i in range(n):
                dists = distances[i].clone()
                dists[i] = float('inf')
                topk_vals, _ = torch.topk(dists, k=n - f - 2, largest=False)
                scores.append(topk_vals.sum().item())

            selected_idx = int(np.argmin(scores))
            selected_update = updates[selected_idx]
            selected_cid = client_ids[selected_idx]

            if selected_cid in self.byzantine_set:
                self.byzantine_selection_count += 1
                print(f"    ❗ 簇内Krum选中拜占庭节点 {selected_cid}（累计: {self.byzantine_selection_count}）")
            else:
                print(f"    ✅ 簇内Krum选择客户端: {selected_cid}")

            return selected_update
        else:
            print(f"    ⚠️ 簇内客户端太少 ({n} < {2 * f + 2})，使用平均 Δw")
            return self._simple_average(updates)

    def aggregate_with_clustered_krum(self, f: int = 1) -> Dict[str, torch.Tensor]:
        """
        新的聚合策略：
        1. 在每个簇内使用Krum选择最可信的更新
        2. 在簇间使用平均聚合得到全局模型
        """
        if not self.client_info or self.global_weights is None:
            print("⚠️ 无客户端数据或全局权重，跳过聚合")
            return self.global_weights

        # 步骤1：按簇分组客户端更新
        cluster_updates = defaultdict(list)
        cluster_client_ids = defaultdict(list)

        for cid, info in self.client_info.items():
            if info and info.get('weight_update') is not None:
                cluster_id = info.get('cluster_id', 0)
                update = info['weight_update']

                # 确保所有更新都在同一设备上
                if len(cluster_updates[cluster_id]) == 0 and update:
                    target_device = next(iter(update.values())).device

                aligned_update = {k: v.to(target_device) for k, v in update.items()}
                cluster_updates[cluster_id].append(aligned_update)
                cluster_client_ids[cluster_id].append(cid)

        if not cluster_updates:
            return self.global_weights

        # 步骤1.5：计算并打印每个簇内客户端本地模型与全局模型的相似度
        print(f"  🔄 开始分层聚合：{len(cluster_updates)} 个簇")

        # 计算相似度的辅助函数 - 计算本地模型权重与全局模型权重的相似度
        def compute_model_similarity(local_weights, global_weights):
            try:
                # 展平本地模型权重
                local_vec = torch.cat([v.flatten() for v in local_weights.values()])

                # 展平全局模型权重
                global_vec = torch.cat([v.flatten() for v in global_weights.values()])

                # 计算余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(
                    local_vec.unsqueeze(0),
                    global_vec.unsqueeze(0),
                    dim=1
                )
                return cos_sim.item()
            except:
                return 0.0

        # 对每个簇计算相似度
        for cluster_id in sorted(cluster_updates.keys()):
            client_ids = cluster_client_ids[cluster_id]
            similarities = []
            byzantine_similarities = []
            normal_similarities = []

            for cid in client_ids:
                local_weights = self.client_info[cid].get('local_weights')
                if local_weights is not None and self.global_weights is not None:
                    # 确保张量在相同设备上
                    local_weights_cpu = {k: v.cpu() for k, v in local_weights.items()}
                    global_weights_cpu = {k: v.cpu() for k, v in self.global_weights.items()}

                    # 计算本地模型与全局模型的相似度
                    sim = compute_model_similarity(local_weights_cpu, global_weights_cpu)
                    similarities.append(sim)

                    if cid in self.byzantine_set:
                        byzantine_similarities.append((cid, sim))
                    else:
                        normal_similarities.append((cid, sim))

            if similarities:
                avg_sim = np.mean(similarities)
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)

                print(f"    📊 簇 {cluster_id} 模型相似度统计:")
                print(f"      平均值: {avg_sim:.4f}, 最小值: {min_sim:.4f}, 最大值: {max_sim:.4f}")

                # 打印正常客户端
                if normal_similarities:
                    print(f"      正常客户端 ({len(normal_similarities)}个): ", end="")
                    for cid, sim in normal_similarities:
                        print(f"c{cid}:{sim:.3f} ", end="")
                    print()

                # 打印拜占庭客户端
                if byzantine_similarities:
                    print(f"      拜占庭客户端 ({len(byzantine_similarities)}个): ", end="")
                    for cid, sim in byzantine_similarities:
                        print(f"c{cid}:{sim:.3f} ", end="")
                    print()

        # 步骤2：在每个簇内使用Krum选择可信更新
        cluster_selected_updates = []

        for cluster_id, updates in cluster_updates.items():
            client_ids = cluster_client_ids[cluster_id]
            print(f"    📊 处理簇 {cluster_id}: {len(updates)} 个客户端")

            selected_update = self._krum_select(updates, client_ids, f)
            if selected_update is not None:
                cluster_selected_updates.append(selected_update)

        # 步骤3：在簇间使用平均聚合
        if not cluster_selected_updates:
            print("⚠️ 所有簇都无有效更新，保持原权重")
            return self.global_weights

        print(f"  🔄 簇间聚合：{len(cluster_selected_updates)} 个簇的更新")
        final_delta = self._simple_average(cluster_selected_updates)

        # 步骤4：应用更新到全局权重
        new_weights = {
            key: self.global_weights[key] + final_delta[key]
            for key in self.global_weights
        }

        print(f"  ✅ 分层聚合完成")
        return new_weights