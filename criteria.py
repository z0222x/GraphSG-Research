import torch
import torch.nn as nn
import torch.nn.functional as F

class NewFrameworkCriterion(nn.Module):
    def __init__(self, tau_cl=1, alpha=1, beta=1, gamma=1, delta=1, task_type='binary', num_classes=2):
        super().__init__()
        self.tau_cl = tau_cl  
        self.alpha = alpha    
        self.beta = beta      
        self.gamma = gamma    
        self.delta = delta    
        self.task_type = task_type  
        self.num_classes = num_classes  

  k2, data):
        total_loss = 1.0
        batch_size = len(g0_node_list)
        eps = 1e-8
        temperature = 0.8

        
       
       
        batch_indices = data.batch.cpu().numpy()
       
        graph_node_ranges = []
        for b in range(batch_size):
            
            node_idx = torch.where(data.batch == b)[0]   #
            if len(node_idx) == 0:
                graph_node_ranges.append((0, 0))
                continue
            start = node_idx[0].item()
            end = node_idx[-1].item() + 1  
            graph_node_ranges.append((start, end))

        
        for b in range(batch_size):
           
            g0_node_b = g0_node_list[b] 
            g1_node_b = g1_node_list[b]  
            g2_node_b = g2_node_list[b]  

            
            start, end = graph_node_ranges[b]
           
            if mask1 is not None and start < end:
                mask1_b = mask1[start:end]
                
                g0_node_b_valid = g0_node_b[mask1_b.bool()]
           
           
            if mask2 is not None and start < end:
                mask2_b = mask2[start:end]
                g0_node_b_valid_2 = g0_node_b[mask2_b.bool()]
           
           
            g0_1 = g0_node_b_valid
            g1_1 = g1_node_b

           
            g0_2 = g0_node_b_valid_2
            g2_2 = g2_node_b


            g0_1 = F.normalize(g0_1, dim=-1)
            g1_1 = F.normalize(g1_1, dim=-1)
            g0_2 = F.normalize(g0_2, dim=-1)
            g2_2 = F.normalize(g2_2, dim=-1)

           
            sim_01 = F.cosine_similarity(g0_1, g1_1, dim=-1)  # [min_len_1,]
            sim_02 = F.cosine_similarity(g0_2, g2_2, dim=-1)  # [min_len_2,]

           
            exp_sim_01 = torch.exp(sim_01 / temperature)
            exp_sim_02 = torch.exp(sim_02 / temperature)
            min_len = min(len(exp_sim_01), len(exp_sim_02))
            exp_sim_01 = exp_sim_01[:min_len]
            exp_sim_02 = exp_sim_02[:min_len]
            loss_01 = -torch.log(exp_sim_01 / (exp_sim_01 + exp_sim_02 + eps) + eps)
            loss_02 = -torch.log(exp_sim_02 / (exp_sim_01 + exp_sim_02 + eps) + eps)

           
            total_loss += (loss_01.mean() + loss_02.mean()) / 2  

        avg_loss = total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=g0_node_list[0].device)
        return avg_loss



    def edge_consistency_loss(self, data):
        total_loss_1 = 0.0
        total_loss_2 = 0.0
        batch_size = data.num_graphs
        device = data.x.device if data.x is not None else torch.device("cpu")

        for b in range(batch_size):
            
            node_mask = data.batch == b
            
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
           
            ori_edge_idx = data.edge_index[:, edge_mask]
            
            aug1_mask = data.mask1[node_mask] 
            aug1_nodes = torch.where(aug1_mask)[0]
            ori_adj_vec_1 = torch.zeros(len(aug1_nodes), len(aug1_nodes), device=device)

            for i, node in enumerate(aug1_nodes):
               
                global_node = torch.where(node_mask)[0][node]
                neighbors = ori_edge_idx[1][ori_edge_idx[0] == global_node]
                
                local_neighbors = torch.where(torch.isin(torch.where(node_mask)[0], neighbors))[0]
                neighbor_idx = torch.where(torch.isin(aug1_nodes, local_neighbors))[0]
                ori_adj_vec_1[i, neighbor_idx] = 1.0

           
            aug1_edge_mask = node_mask[data.edge_index1[0]] & node_mask[data.edge_index1[1]]
            aug1_edge_idx = data.edge_index1[:, aug1_edge_mask]
           

            aug1_adj_vec = torch.zeros(len(aug1_nodes), len(aug1_nodes), device=device)
            for i, node in enumerate(aug1_nodes):
                global_node = torch.where(node_mask)[0][node]
                neighbors = aug1_edge_idx[1][aug1_edge_idx[0] == global_node]
                local_neighbors = torch.where(torch.isin(torch.where(node_mask)[0], neighbors))[0]
                neighbor_idx = torch.where(torch.isin(aug1_nodes, local_neighbors))[0]
                aug1_adj_vec[i, neighbor_idx] = 1.0

            loss_1 = F.mse_loss(ori_adj_vec_1, aug1_adj_vec)
            total_loss_1 += loss_1

           
            aug2_mask = data.mask2[node_mask]
            aug2_nodes = torch.where(aug2_mask)[0]
            ori_adj_vec_2 = torch.zeros(len(aug2_nodes), len(aug2_nodes), device=device)

            for i, node in enumerate(aug2_nodes):
                global_node = torch.where(node_mask)[0][node]
                neighbors = ori_edge_idx[1][ori_edge_idx[0] == global_node]
                local_neighbors = torch.where(torch.isin(torch.where(node_mask)[0], neighbors))[0]
                neighbor_idx = torch.where(torch.isin(aug2_nodes, local_neighbors))[0]
                ori_adj_vec_2[i, neighbor_idx] = 1.0

            
            aug2_edge_mask = node_mask[data.edge_index2[0]] & node_mask[data.edge_index2[1]]
            aug2_edge_idx = data.edge_index2[:, aug2_edge_mask]
            
            aug2_adj_vec = torch.zeros(len(aug2_nodes), len(aug2_nodes), device=device)
            for i, node in enumerate(aug2_nodes):
                global_node = torch.where(node_mask)[0][node]
                neighbors = aug2_edge_idx[1][aug2_edge_idx[0] == global_node]
                local_neighbors = torch.where(torch.isin(torch.where(node_mask)[0], neighbors))[0]
                neighbor_idx = torch.where(torch.isin(aug2_nodes, local_neighbors))[0]
                aug2_adj_vec[i, neighbor_idx] = 1.0

            loss_2 = F.mse_loss(ori_adj_vec_2, aug2_adj_vec)
            total_loss_2 += loss_2

        avg_loss_1 = total_loss_1 / batch_size
        avg_loss_2 = total_loss_2 / batch_size
        return (avg_loss_1 + avg_loss_2) / 2

   
    def task_correlation_loss(self, G_star_list, g0_graph_list, g1_graph_list, g2_graph_list):
        total_loss = 1.0
        batch_size = len(G_star_list)
        eps = 1e-8  # 数值稳定项，避免除0/负数

        
      
        
        for b in range(batch_size):
            # 第b张图的G*和g0/g1/g2（严格对应）
            g_star = G_star_list[b]  # 第b张图的G* (256,)
            g0 = g0_graph_list[b]  # 第b张图的g0 (256,)
            g1 = g1_graph_list[b]  # 第b张图的g1 (256,)
            g2 = g2_graph_list[b]  # 第b张图的g2 (256,)

            g_star_centered = g_star - g_star.mean()
            g0_centered = g0 - g0.mean()
            g1_centered = g1 - g1.mean()
            g2_centered = g2 - g2.mean()

            # 皮尔逊相关系数计算（适配256维向量）
            def corr(vec_a, vec_b):
                numerator = torch.dot(vec_a, vec_b)
                denominator = torch.norm(vec_a, p=2) * torch.norm(vec_b, p=2) + eps
                return numerator / denominator

            # 计算第b张图的相关性
            corr0 = corr(g_star_centered, g0_centered)
            corr1 = corr(g_star_centered, g1_centered)
            corr2 = corr(g_star_centered, g2_centered)

            # 数值稳定：限制相关性最小值为eps（避免负数/0导致损失爆炸）
            corr0 = torch.clamp(corr0, min=eps)
            corr1 = torch.clamp(corr1, min=eps)
            corr2 = torch.clamp(corr2, min=eps)

           
            loss_b = 1 / (corr0 + eps) + 1 / (corr1 + eps) + 1 / (corr2 + eps)
            total_loss += loss_b

        # 计算batch平均损失
        return total_loss / batch_size

    
    def forward(
            self, pred_task, y,
            g0_node_list, g1_node_list, g2_node_list,  # 节点级批量列表
            g0_graph_list, g1_graph_list, g2_graph_list,  # 图级批量列表
            mask1, mask2, G_star_list, data
    ):
        # 1. 分类损失（原有逻辑，适配批量）
        if self.num_classes == 2:
            # 二分类：pred_task是[B,1]，y是[B] → 调整维度匹配
            ce_loss = F.binary_cross_entropy_with_logits(pred_task.squeeze(1), y.float())
        else:
            # 多分类：pred_task是[B, num_classes]，y是[B]
            ce_loss = F.cross_entropy(pred_task, y)

        # 2. 节点对比损失（传入批量节点列表）
        node_cl_loss = self.node_contrastive_loss(g0_node_list, g1_node_list, g2_node_list, mask1, mask2, data)

        # 3. 相关性损失（传入批量图级列表，一一对应）
        corr_loss = self.task_correlation_loss(G_star_list, g0_graph_list, g1_graph_list, g2_graph_list)

        # 4. 边一致性损失（原有逻辑，适配批量）
        edge_cons_loss = self.edge_consistency_loss(data)

        # 5. 总损失（权重可根据需求调整）
        total_loss = ce_loss + 0.1 * node_cl_loss + 0.1 * corr_loss + 0.1 * edge_cons_loss

        # 记录各损失值（方便调试）
        loss_dict = {
            "ce_loss": ce_loss.item(),
            "node_cl_loss": node_cl_loss.item(),
            "corr_loss": corr_loss.item(),
            "edge_cons_loss": edge_cons_loss.item()
        }
        return total_loss, loss_dict

class InfoGuideCriterion(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, task_type='binary', num_classes=2):
        super().__init__()
        self.alpha = alpha  # 对齐损失权重（优先保证G*对齐）
        self.beta = beta    # 分类损失权重（保证分类性能）
        self.task_type = task_type  # 任务类型
        self.num_classes = num_classes  # 分类数

    # 对齐损失：保证信息指导模块的输出与G*的一致性

    def alignment_loss(self, g_proj, G_star_node_list, data):
      
        total_loss = 0.0
        batch_size = data.num_graphs
        device = g_proj.device
        batch = data.batch.to(device) if hasattr(data, 'batch') else torch.zeros(
            g_proj.shape[0], dtype=torch.long, device=device
        )

        for b in range(batch_size):
            
            mask = batch == b
            if not torch.any(mask):
                continue

            
            gp_nodes = g_proj[mask]

           
            g_star = G_star_node_list[b].to(device)  

            
            node_wise_loss = torch.norm(gp_nodes - g_star, p=2, dim=1) ** 2
            graph_align_loss = node_wise_loss.mean()  

            total_loss += graph_align_loss

        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=device)


   
    def forward(self, g_proj, pred, target, G_star_node_list, data):
        # 1. 对齐损失
        align_loss = self.alignment_loss(g_proj, G_star_node_list, data)
        # 2. 分类损失（适配任务类型）
        if self.task_type == 'binary':
            ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), target.float())
        elif self.task_type == 'multiclass':
            ce_loss = F.cross_entropy(pred, target.long())
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        # 总损失（加权求和）
        total_loss = self.alpha * align_loss + self.beta * ce_loss
        return total_loss, {'align_loss': align_loss, 'ce_loss': ce_loss}
