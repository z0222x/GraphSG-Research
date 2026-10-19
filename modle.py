import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data



class LightGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv = GCNConv(input_dim, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x, edge_index, data):
       
        x = F.relu(self.conv(x, edge_index))
        return self.fc(x)


#任务信息提取模块的GNN（GNN0/GNN1/GNN2共用此架构）
#GCN
class TaskGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

  
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, data):
     
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_index.numel() > 0:
            mask = (edge_index[0] < x.shape[0]) & (edge_index[1] < x.shape[0])
            edge_index = edge_index[:, mask]
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                
            max_node_idx = edge_index.max().item()
           
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

    
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)  

      
        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2 + x1)  # 残差缓解梯度消失
        x2 = F.dropout(x2, p=0.4, training=self.training) 

      
        x3 = self.conv3(x2, edge_index)
        x3 = self.norm3(x3)

        return x3  # 返回节点级特征





class NonLinearMLP(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.mlp(x)




# 任务信息提取模块（核心模块：完全复现框架G*提取逻辑）
class TaskInfoExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, task_dim=256, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim  
        
        self.task_dim = task_dim  
        self.num_classes = num_classes 
        self.rcov = 1e-8  
        self.viewWts = [1.0, 1.0, 1.0]  
        self.max_r = 12

        self.feat_down = nn.Linear(256, self.max_r)

        # 1. 三个视图的GNN（原始图G0、增强图G1、增强图G2）
        self.gnn0 = TaskGNN(input_dim, hidden_dim, hidden_dim)
        self.gnn1 = TaskGNN(input_dim, hidden_dim, hidden_dim)
        self.gnn2 = TaskGNN(input_dim, hidden_dim, hidden_dim)

        # 2. 非线性映射MLP（每个视图独立，框架要求256→128→64）
        self.mlp0 = NonLinearMLP(hidden_dim)
        self.mlp1 = NonLinearMLP(hidden_dim)
        self.mlp2 = NonLinearMLP(hidden_dim)

        # 3. DGCCA投影矩阵（随机正交初始化，框架要求Uj为64×r维度）
        self.U0 = nn.Parameter(self._orthogonal_init(64, self.max_r))  # 64×64
        self.U1 = nn.Parameter(self._orthogonal_init(64, self.max_r))
        self.U2 = nn.Parameter(self._orthogonal_init(64, self.max_r))
        self.U = nn.Parameter(self._orthogonal_init(self.max_r, self.task_dim))  # 8×256

        # 4. 分类头（框架要求2~3层MLP）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # 二分类输出1维（logits），多分类输出num_classes维
            nn.Linear(64, 1 if self.num_classes == 2 else self.num_classes)
        )

    def _orthogonal_init(self, in_dim, out_dim):
        """用官方正交初始化，严格保证维度"""
        # 1创建指定维度的空矩阵
        mat = torch.empty(in_dim, out_dim)
        # 官方正交初始化（自动处理奇异矩阵，维度绝对不丢）
        nn.init.orthogonal_(mat)
        # 强制返回指定维度（兜底）
        assert mat.shape == (in_dim, out_dim), f"初始化维度错误：{mat.shape}≠({in_dim},{out_dim})"
        return mat


    def forward(self, data):
       
      
        device = next(self.parameters()).device
        data = data.to(device)

       
        g0_feat = self.gnn0(data.x, data.edge_index, data)  # [N0_total, 256]
        N0_total = g0_feat.shape[0]  # 原始图总节点数


       
        aug1_data = Data(
            x=data.x1.to(device),
            edge_index=data.edge_index1.to(device),
            batch=data.x1_batch.to(device) if hasattr(data, 'x1_batch') else torch.zeros(data.x1.shape[0],
                                                                                         dtype=torch.long).to(
                device)
        )
        g1_feat = self.gnn1(aug1_data.x, aug1_data.edge_index, aug1_data)  # [N1_total, 256]


   
        aug2_data = Data(
            x=data.x2.to(device),
            edge_index=data.edge_index2.to(device),
            batch=data.x2_batch.to(device) if hasattr(data, 'x2_batch') else torch.zeros(data.x2.shape[0],
                                                                                         dtype=torch.long).to(
                device)
        )
        g2_feat = self.gnn2(aug2_data.x, aug2_data.edge_index, aug2_data)  # [N2_total, 256]


        
        batch = data.batch  
        batch_size = batch.max().item() + 1 if (batch.numel() > 0 and batch.max().item() >= 0) else 1

       
        g0_ranges = []  # [(start0, end0), (start1, end1), ...]
        for b in range(batch_size):
            graph_mask = batch == b
            if graph_mask.any():
                start = torch.where(graph_mask)[0][0].item()
                end = torch.where(graph_mask)[0][-1].item() + 1  # 左闭右开
                g0_ranges.append((start, end))
                node_num = end - start
            else:
                g0_ranges.append((0, 0))


        
        g0_node_list = []  # 节点级：[图1节点特征, 图2节点特征, ...]
        g1_node_list = []
        g2_node_list = []
        g0_graph_list = []  # 图级：[图1g0, 图2g0, ...]
        g1_graph_list = []
        g2_graph_list = []
        G_star_node_list = []  # 节点级G*列表
        G_star_list = []  # 图级G*列表

        # 增强图累积索引（仅初始化一次）
        g1_cumulative = 0
        g2_cumulative = 0

        # 图级特征计算函数（仅用增强图真实节点数，无需补0）
        def get_graph_level_feat(node_feat):
            # if node_feat is None or node_feat.shape[0] == 0:
            #     return torch.zeros(256, device=device)
            return node_feat.mean(dim=0)

        # ---------------------- 单次遍历：提取特征 + 计算G* ----------------------
        for b in range(batch_size):
            g0_start, g0_end = g0_ranges[b]
            graph_mask = batch == b


            # ========== 第一步：提取第b张图的原始图/增强图特征 ==========
            # 原始图特征
            g0_node_b = g0_feat[g0_start:g0_end]  # [N0_b, 256] 真实节点数，无补0
            N0_b = g0_node_b.shape[0]  # 单图原始图节点数
            g0_graph_b = get_graph_level_feat(g0_node_b)  # [256,]


            # 增强图1特征
            n1_b = data.num_nodes1[b].item()  # 第b张图增强图1的真实节点数
            g1_node_b = g1_feat[g1_cumulative: g1_cumulative + n1_b]  # [N1_b, 256]
            g1_graph_b = get_graph_level_feat(g1_node_b)
            # 仅记录累积索引（不更新！更新放在G*计算后，避免重复累加）
            g1_cumulative_temp = g1_cumulative + n1_b

            # 增强图2特征
            n2_b = data.num_nodes2[b].item()  # 第b张图增强图2的真实节点数
            g2_node_b = g2_feat[g2_cumulative: g2_cumulative + n2_b]  # [N2_b, 256]
            g2_graph_b = get_graph_level_feat(g2_node_b)
            g2_cumulative_temp = g2_cumulative + n2_b


            # 保存特征列表
            g0_node_list.append(g0_node_b)
            g1_node_list.append(g1_node_b)
            g2_node_list.append(g2_node_b)
            g0_graph_list.append(g0_graph_b)
            g1_graph_list.append(g1_graph_b)
            g2_graph_list.append(g2_graph_b)

           
            if g1_node_b.shape[0] < N0_b:
                pad1 = torch.zeros(N0_b - g1_node_b.shape[0], g1_node_b.shape[1], device=device)
                g1_node_b_padded = torch.cat([g1_node_b, pad1], dim=0)
           


            # 增强图2补0到原始图维度
            if g2_node_b.shape[0] < N0_b:
                pad2 = torch.zeros(N0_b - g2_node_b.shape[0], g2_node_b.shape[1], device=device)
                g2_node_b_padded = torch.cat([g2_node_b, pad2], dim=0)
           


            # 2. 单图非线性映射（MLP）
            f0_b = self.mlp0(g0_node_b)  # (N0_b, 64)
            f1_b = self.mlp1(g1_node_b_padded)  # (N0_b, 64)
            f2_b = self.mlp2(g2_node_b_padded)  # (N0_b, 64)

            #特征归一化
            f0_b = f0_b / (torch.norm(f0_b, p=2, dim=1, keepdim=True) + 1e-8)
            f1_b = f1_b / (torch.norm(f1_b, p=2, dim=1, keepdim=True) + 1e-8)
            f2_b = f2_b / (torch.norm(f2_b, p=2, dim=1, keepdim=True) + 1e-8)

            # 3. 单图特征中心化
            def center_feature(f):

                f = torch.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)

                mean = f.mean(dim=0, keepdim=True)
                return f - mean

            f0_centered_b = center_feature(f0_b)
            f1_centered_b = center_feature(f1_b)
            f2_centered_b = center_feature(f2_b)

            
            # 4. 单图动态计算r值
            min_nodes_b = N0_b
            min_nodes_b = max(min_nodes_b, 2)
            if min_nodes_b <= 100:
                r_b = min_nodes_b // 2
            else:
                r_b = min_nodes_b // 3
            r_b = max(r_b, 3)
            r_b = min(r_b, self.max_r)
            assert 1 <= r_b <= self.max_r, f"单图{r_b}值超出合法范围[1, {self.max_r}]"

            # Z0_b = self.feat_down(raw0)[:, :r_b]  # (N0_b, r_b)
            # Z1_b = self.feat_down(raw1)[:, :r_b]  # (N0_b, r_b)
            # Z2_b = self.feat_down(raw2)[:, :r_b]  # (N0_b, r_b)

            # 5. 单图调整投影矩阵维度
            U0_b = self.U0[:, :r_b].to(device)
            U1_b = self.U1[:, :r_b].to(device)
            U2_b = self.U2[:, :r_b].to(device)
            U_b = self.U[:r_b, :].to(device)
            # U_b = self.U[:r_b, :].to(device)


            # 6. 单图共享空间投影
            Z0_b = torch.matmul(f0_centered_b, U0_b)  # (N0_b, r_b)
            Z1_b = torch.matmul(f1_centered_b, U1_b)  # (N0_b, r_b)
            Z2_b = torch.matmul(f2_centered_b, U2_b)  # (N0_b, r_b)

            Z0_b = torch.clamp(Z0_b, min=-1e4, max=1e4)
            Z1_b = torch.clamp(Z1_b, min=-1e4, max=1e4)
            Z2_b = torch.clamp(Z2_b, min=-1e4, max=1e4)



            # 7. 单图协方差矩阵计算
            C0_b = torch.matmul(Z0_b.T, Z0_b)+ 1e-8 * torch.eye(r_b, device=device)  # (r_b, r_b)
            C1_b = torch.matmul(Z1_b.T, Z1_b) + 1e-8 * torch.eye(r_b, device=device) # (r_b, r_b)
            C2_b = torch.matmul(Z2_b.T, Z2_b) + 1e-8 * torch.eye(r_b, device=device) # (r_b, r_b)

          
            def regularized_inverse(self, C):
                # 1. 先加一层微小的对角阻尼，保持原有分布
                I = torch.eye(C.size(0), device=C.device)
                C_reg = C + self.rcov * I

                # 2. 放弃 torch.inverse，改用 pinv（伪逆）
                # rcond=1e-15 表示绝对值小于 1e-15 的奇异值都会被视为0截断，绝不报错
                return torch.linalg.pinv(C_reg, rcond=1e-15)

            for idx, Z in enumerate([Z0_b, Z1_b, Z2_b], 0):
                if torch.isnan(Z).any():
                    raise ValueError(
                        f"投影矩阵Z[{idx}]_b包含NaN！形状: {Z.shape}, 数值预览:\n{Z[:5, :5].detach().cpu().numpy() if Z.shape[0]>=5 else Z.detach().cpu().numpy()}")
                if torch.isinf(Z).any():
                    raise ValueError(
                        f"投影矩阵Z{idx}_b包含Inf！形状: {Z.shape}, 数值预览:\n{Z[:5, :5].detach().cpu().numpy() if Z.shape[0] >= 5 else Z.detach().cpu().numpy()}")

            C0_inv_b = regularized_inverse(self,C0_b)
            C1_inv_b = regularized_inverse(self,C1_b)
            C2_inv_b = regularized_inverse(self,C2_b)

            # 9. 单图白化投影矩阵
            P0_b = torch.matmul(torch.matmul(Z0_b, C0_inv_b), Z0_b.T)  # (N0_b, N0_b)
            P1_b = torch.matmul(torch.matmul(Z1_b, C1_inv_b), Z1_b.T)  # (N0_b, N0_b)
            P2_b = torch.matmul(torch.matmul(Z2_b, C2_inv_b), Z2_b.T)  # (N0_b, N0_b)

         
            # print(f"\n===== 计算融合矩阵M_b =====")
            # 10. 单图融合矩阵
            M_b = self.viewWts[0] * P0_b + self.viewWts[1] * P1_b + self.viewWts[2] * P2_b  # (N0_b, N0_b)
            M_b = M_b + 1e-6 * torch.eye(M_b.shape[0], device=M_b.device)

            
            
            # 11. 单图SVD分解（利用自动降序特性）
            U, S, V = torch.linalg.svd(M_b, full_matrices=False)  # 推荐加full_matrices=False

            # 12. 直接取前r_b个奇异向量（无需排序！）
            Q_r_b = U[:, :r_b]  # (N0_b, r_b) 维度和原逻辑完全一致

            # 13. 单图维度匹配得到G*
            G_star_b = torch.matmul(Q_r_b, U_b)  # (N0_b, 256)
            
            # M_b = self.viewWts[0] * Z0_b + self.viewWts[1] * Z1_b + self.viewWts[2] *  Z2_b
            # G_star_b=torch.matmul(M_b , U_b)


            # 14. 单图图级G*（均值池化）
            G_avg_b = G_star_b.mean(dim=0)  # (256,)

            # 15. 保存G*列表
            G_star_node_list.append(G_star_b)
            G_star_list.append(G_avg_b)

            # 16. 更新增强图累积索引（仅在非空图时更新）
            g1_cumulative = g1_cumulative_temp
            g2_cumulative = g2_cumulative_temp

       
        # 分类预测
        G_star_batch = torch.stack(G_star_list) 
        pred = self.classifier(G_star_batch)  # 二分类：(B,1)；多分类：(B, num_classes)

        return pred, g0_node_list, g1_node_list, g2_node_list, g0_graph_list, g1_graph_list, g2_graph_list, G_star_list, G_star_node_list

      
      
      class InfoGuideModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, task_dim=256, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.num_classes = num_classes
        
        self.gnn = TaskGNN(input_dim, hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(64, 1 if self.num_classes == 2 else self.num_classes)
        )



    def forward(self, data, G_star_node_list=None):
       
        feat = self.gnn(data.x, data.edge_index, data)  # (N, 256)

        if G_star_node_list is not None:
            
            return feat
        else:
            
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(feat.shape[0], dtype=torch.long).to(
                feat.device)
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
            pred_list = []
            for b in range(batch_size):
                mask = batch == b
                gp = feat[mask]
               
                w = torch.ones(gp.shape[0], device=gp.device)
                if hasattr(data, 'original_node_mask'):
                    w[data.original_node_mask[mask] == 0] = 0
               
                gp_pooled = (gp.T @ w) / (w.sum() + 1e-8)  # (256,)
                pred = self.classifier(gp_pooled)
                pred_list.append(pred)
            return torch.stack(pred_list)