import os
import pickle
import numpy as np
import torch
# from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import InMemoryDataset, Data as PyGData


# 上采样数据集类（适配二分类/多分类，解决类别不平衡）
class UpsampledDataset(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, dataset_name, task_type='binary', num_classes=2,
                 structural_feature_name="ori", transform=None, pre_transform=None, pre_filter=None,min_node_num=8):
        self.cur_dataset = cur_dataset  # 原始数据集
        self.dataset_name = dataset_name  # 数据集名称
        self.task_type = task_type  # 任务类型（binary/multiclass）
        self._num_classes = num_classes  # 分类数
        self.min_node_num = min_node_num  # 最小节点数要求
        # 定义数据集根目录
        if structural_feature_name == "ori":
            root_name = dataset_root + "_ori_upsample_train"
        else:
            raise NotImplementedError("Only support 'ori' structural feature for new framework")
        root = os.path.join(root_name, "train")
        super().__init__(root, transform, pre_transform, pre_filter)
        # 加载处理后的数据集（data：批量PyG Data对象；slices：切片索引用于批量加载）
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_file_names(self):
        return ['graph_data.pt']  # 处理后的文件名

    def process(self):


        data_list = []
        # 二分类数据集：按数据集名称执行不同的上采样策略
        if self.task_type == 'binary':
            for idx, sample in enumerate(self.cur_dataset):
                curr_y = sample.y.item()
                if self.dataset_name == "bbbp":
                    if curr_y == 1:
                        data_list.append(sample)
                    elif curr_y == 0:
                        for _ in range(6):  # 负样本上采样6倍
                            data_list.append(sample)
                elif self.dataset_name == "bace":
                    if curr_y == 0:
                        data_list.append(sample)
                    elif curr_y == 1:
                        for _ in range(2):  # 正样本上采样2倍
                            data_list.append(sample)
                elif self.dataset_name == "hiv":
                    if curr_y == 0:
                        data_list.append(sample)
                    elif curr_y == 1:
                        for _ in range(20):  # 正样本上采样20倍
                            data_list.append(sample)
                else:
                    # 其他二分类数据集：不执行上采样
                    data_list.append(sample)
        else:
            # 多分类数据集：暂不执行上采样（可根据需求自定义）
            data_list = self.cur_dataset.copy()
        # 应用数据过滤（如过滤节点数过少的图）
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        # 应用数据变换（如归一化、加自环等）
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        # 合并数据集并保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class AugmentedDatasetNew(InMemoryDataset):
    def __init__(self, cur_dataset, dataset_root, split_name, task_type='binary', num_classes=2,
                 sample_ratios=[0.15, 0.3], device=None, transform=None, pre_transform=None, pre_filter=None):
        self.cur_dataset = cur_dataset  
        self.sample_ratios = sample_ratios  
        self.device = device  
        self.task_type = task_type  
        self._num_classes = num_classes  
        
        root_name = dataset_root + "_augmented_new"
        root = os.path.join(root_name, split_name)
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_file_names(self):
        return ['graph_data.pt']


    def process(self):
        data_list = [] 

        if self.cur_dataset[0].x is None:
            
            unlocked_dataset = [] 

            for sample in self.cur_dataset:
              
                if hasattr(sample, 'num_nodes') and sample.num_nodes is not None:
                    num_nodes = sample.num_nodes
                elif sample.edge_index is not None and sample.edge_index.numel() > 0:
                    num_nodes = int(sample.edge_index.max().item() + 1)
               

                
                dummy_x = torch.ones((num_nodes, 1), dtype=torch.float32)

                
                from torch_geometric.data import Data
                new_sample = Data(
                    x=dummy_x,  
                    edge_index=sample.edge_index,  
                    y=sample.y,  # 完美继承原图的标签
                    num_nodes=num_nodes  
                )

               
                for key in sample.keys():
                    if key not in ['x', 'edge_index', 'y', 'num_nodes']:
                        setattr(new_sample, key, sample[key])

                unlocked_dataset.append(new_sample)

            
            self.cur_dataset = unlocked_dataset

        
        light_gnn = LightGNN(input_dim=self.cur_dataset[0].x.shape[-1], hidden_dim=128).to(self.device)
        light_gnn.eval()  


        for sample in self.cur_dataset:
            sample = sample.to(self.device)  
            sample.x = sample.x.float()  
           
            if self.task_type == 'binary':
                sample_y = sample.y.float().squeeze() 
            elif self.task_type == 'multiclass':
                sample_y = sample.y.long().squeeze()  

           
            node_repr = light_gnn(sample.x, sample.edge_index, sample)
            z_v = torch.sigmoid(node_repr).squeeze().detach().cpu().numpy()

           
            z_v_min = z_v.min()
            z_v_max = z_v.max()
           
            if z_v_max - z_v_min < 1e-6:
                z_v = np.linspace(0.0, 1.0, len(z_v)) 
            else:
                z_v = (z_v - z_v_min) / (z_v_max - z_v_min)  
            # ======================================================================

           
            original_data = PyGData(
                x=sample.x.cpu(),  # 原始节点特征
                edge_index=sample.edge_index.cpu(),  # 原始边索引
                y=sample_y.cpu(),  # 统一格式的标签
                is_original=True,  # 标记是原始图（区分增强图）
                num_nodes=sample.x.shape[0]  # 原始图节点数
            )

           
            aug_count = 0  # 增强图计数（0=增强图1，1=增强图2）
            for r in self.sample_ratios:  # sample_ratios是增强比例列表
               
                sorted_idx = np.argsort(z_v)  
                threshold_idx = int(r * len(z_v))  

               
                threshold_idx = max(threshold_idx, 1)  
                threshold_idx = min(threshold_idx, len(z_v) - 2)  
               
                theta_r = z_v[sorted_idx[threshold_idx]]

                
                tau = 0.5  
                eps0 = torch.randn_like(torch.tensor(z_v)) * 0.01  # 高斯噪声（增加鲁棒性）
                eps1 = torch.randn_like(torch.tensor(z_v)) * 0.01

                
                logits = (torch.log(torch.tensor(z_v) + 1e-8) - torch.log(1 - torch.tensor(z_v) + 1e-8)
                          + (eps0 - eps1) - torch.log(torch.tensor(theta_r) + 1e-8) + torch.log(
                            1 - torch.tensor(theta_r) + 1e-8)) / tau
                m = torch.sigmoid(logits).numpy()  #

                
                keep_nodes = np.where(m > 0.5)[0]
                if len(keep_nodes) < 2:
                    keep_nodes = np.where(m > 0.1)[0] if len(np.where(m > 0.1)[0]) >= 2 else np.arange(min(2, len(z_v)))


               
                original_num = len(z_v)
                target_keep_num = int(original_num * (1 - r))  # 目标保留节点数
                target_keep_num = max(target_keep_num, 2)  # 至少保留2个

                
                if len(keep_nodes) >= original_num:
                    # 按z_v降序取前target_keep_num个（重要性高的节点）
                    sorted_by_importance = np.argsort(z_v)[::-1]
                    keep_nodes = sorted_by_importance[:target_keep_num]
                    m = np.zeros_like(m)  
                    m[keep_nodes] = 1.0  
           
                keep_nodes = np.sort(keep_nodes) 

                num_aug_nodes = len(keep_nodes)

                

                def _gumbel_softmax_mask(self, score, theta, tau=0.5):
                   
                    z_v = torch.tensor(score)
                    theta_r = torch.tensor(theta)

                    eps0 = torch.randn_like(z_v) * 0.01
                    eps1 = torch.randn_like(z_v)

                   
                    logits = (torch.log(z_v + 1e-8) - torch.log(1 - z_v + 1e-8)
                              + (eps0 - eps1)
                              - torch.log(theta_r + 1e-8)
                              + torch.log(1 - theta_r + 1e-8)) / tau

                    return torch.sigmoid(logits).numpy()
               
                edge_index = sample.edge_index.cpu().numpy()  
               
                valid_edges = np.logical_and(
                    np.isin(edge_index[0], keep_nodes),  
                    np.isin(edge_index[1], keep_nodes)  
                )
                new_edge_index = edge_index[:, valid_edges]  
                node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}
             
               
               

               
                if new_edge_index.size > 0:
                    new_edge_index_0 = np.vectorize(node_map.get)(new_edge_index[0])
                    new_edge_index_1 = np.vectorize(node_map.get)(new_edge_index[1])
                    new_edge_index = np.array([new_edge_index_0, new_edge_index_1], dtype=np.int64)
                else:
                    new_edge_index = np.array([[], []], dtype=np.int64)

               
                if new_edge_index.size > 0:
                    max_idx = max(new_edge_index[0].max(), new_edge_index[1].max())
                    min_idx = min(new_edge_index[0].min(), new_edge_index[1].min())
                   
             
                new_edge_index = torch.tensor(new_edge_index, dtype=torch.int64)
                
             
                new_x = sample.x[keep_nodes].cpu()

                
                    original_data.x1 = new_x
                    original_data.edge_index1 = new_edge_index
                    original_data.mask1 = torch.tensor(m > 0.5).cpu() 
                    original_data.num_nodes1 = num_aug_nodes
               

                    
                    original_data.x2 = new_x
                    original_data.edge_index2 = new_edge_index
                    original_data.mask2 = torch.tensor(m > 0.5).cpu()  
                    original_data.num_nodes2 = num_aug_nodes

                aug_count += 1

           
            data_list.append(original_data)

       
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def quick_test_data_prep(cur_data, task_type='binary', num_classes=2):
    
    original_cur_data_len = len(cur_data)
  

    
    valid_indices = []
    invalid_sample_info = []  
    for idx, sample in enumerate(cur_data):
       
        if sample.x is not None:
            num_nodes = sample.x.shape[0]
        elif hasattr(sample, 'num_nodes') and sample.num_nodes is not None:
            num_nodes = sample.num_nodes
        elif sample.edge_index is not None and sample.edge_index.numel() > 0:
            num_nodes = int(sample.edge_index.max().item() + 1)
        else:
            num_nodes = 0

        if num_nodes >= 8: 
            valid_indices.append(idx)
        else:
            invalid_sample_info.append(f"第{idx}个图：节点数={num_nodes}")

   
    cur_data = [cur_data[idx] for idx in valid_indices]

   
    
    for i, sample in enumerate(cur_data):
        if sample.x is not None:
            num_nodes = sample.x.shape[0]
        else:
            num_nodes = sample.num_nodes if hasattr(sample, 'num_nodes') else int(sample.edge_index.max().item() + 1)
        

   
    sizes = []
    for sample in cur_data:
        if sample.x is not None:
            sizes.append(sample.x.shape[0])
        else:
            sizes.append(sample.num_nodes if hasattr(sample, 'num_nodes') else int(sample.edge_index.max().item() + 1))
    sizes = np.array(sizes)
    sorted_index = np.argsort(sizes)  
    
    num_test = np.ceil(sizes.shape[0] * 0.10).astype(int)
    num_train_val_test = np.ceil(sizes.shape[0] * 0.50).astype(int)
    large_test_idx = sorted_index[-num_test:]
    train_val_test_idx = sorted_index[:num_train_val_test]

   
    if task_type == 'binary':
        # 二分类：分离正负样本索引
        ones_idx_list, zeros_idx_list = [], []
        for idx in range(len(cur_data)):
            # 🌟🌟🌟 核心修改点：取消对 x.shape[0] != 1 的一刀切限制，允许 x 为 None 时的无特征图进入划分
            if idx in train_val_test_idx:
                y = cur_data[idx].y.item()
                if y == 1:
                    ones_idx_list.append(idx)
                elif y == 0.0 or y == 0:
                    zeros_idx_list.append(idx)
                else:
                    raise ValueError(f"Binary classification expects y=0/1, got {y}")
        ones_idx = np.array(ones_idx_list)
        zeros_idx = np.array(zeros_idx_list)

        
       
        np.random.shuffle(ones_idx)
        np.random.shuffle(zeros_idx)
       
        train_split, val_split, test_split = 0.7, 0.15, 0.15
        num_train_ones = int(len(ones_idx) * train_split)
        num_test_ones = int(len(ones_idx) * test_split)
        num_train_zeros = int(len(zeros_idx) * train_split)
        num_test_zeros = int(len(zeros_idx) * test_split)
        # 分配索引
        train_one_idx = ones_idx[:num_train_ones]
        test_one_idx = ones_idx[num_train_ones:num_train_ones + num_test_ones]
        val_one_idx = ones_idx[num_train_ones + num_test_ones:]
        train_zero_idx = zeros_idx[:num_train_zeros]
        test_zero_idx = zeros_idx[num_train_zeros:num_train_zeros + num_test_zeros]
        val_zero_idx = zeros_idx[num_train_zeros + num_test_zeros:]
        # 合并并打乱索引
        train_idx = np.random.permutation(np.concatenate([train_one_idx, train_zero_idx]))
        val_idx = np.random.permutation(np.concatenate([val_one_idx, val_zero_idx]))
        small_test_idx = np.random.permutation(np.concatenate([test_one_idx, test_zero_idx]))

        # 构建大型测试集（保证类别平衡）
        large_test_idx_recorder = []
        num_large_zeros, num_large_ones = 0, 0
        cap_large_zero = len(test_zero_idx)
        cap_large_one = len(test_one_idx)
        for idx in np.flip(sorted_index):
            if idx not in train_val_test_idx:
                sample = cur_data[idx]
                y = sample.y.item()
                if (y == 0 or y == 0.0) and num_large_zeros < cap_large_zero:
                    large_test_idx_recorder.append(idx)
                    num_large_zeros += 1
                elif y == 1 and num_large_ones < cap_large_one:
                    large_test_idx_recorder.append(idx)
                    num_large_ones += 1
                if num_large_zeros >= cap_large_zero and num_large_ones >= cap_large_one:
                    break
        large_test_idx = np.array(large_test_idx_recorder)
        large_test_idx = np.random.permutation(large_test_idx)
    else:
        # 多分类：按每个类别平衡划分（修改：取消对 x 的严格检查）
        class_indices = {c: [] for c in range(num_classes)}
        for idx in range(len(cur_data)):
            if idx in train_val_test_idx:
                y = cur_data[idx].y.item()
                if 0 <= y < num_classes:
                    class_indices[y].append(idx)
                else:
                    raise ValueError(f"Class {y} out of range [0, {num_classes - 1}]")
        train_idx, val_idx, small_test_idx = [], [], []
        train_split, val_split, test_split = 0.7, 0.15, 0.15
        # 对每个类别单独划分
        for c in range(num_classes):
            indices = np.array(class_indices[c])
            if len(indices) == 0:
                continue
            np.random.shuffle(indices)
            num_train = int(len(indices) * train_split)
            num_test = int(len(indices) * test_split)
            train_idx.extend(indices[:num_train])
            test_idx_c = indices[num_train:num_train + num_test]
            val_idx_c = indices[num_train + num_test:]
            small_test_idx.extend(test_idx_c)
            val_idx.extend(val_idx_c)
        # 打乱索引
        train_idx = np.random.permutation(train_idx)
        val_idx = np.random.permutation(val_idx)
        small_test_idx = np.random.permutation(small_test_idx)

        # 构建大型测试集（多分类）
        large_test_idx_recorder = []
        class_counts = {c: 0 for c in range(num_classes)}
        cap_per_class = len(small_test_idx) // num_classes  # 每个类别分配相同数量的样本
        for idx in np.flip(sorted_index):
            if idx not in train_val_test_idx:
                y = cur_data[idx].y.item()
                if 0 <= y < num_classes and class_counts[y] < cap_per_class:
                    large_test_idx_recorder.append(idx)
                    class_counts[y] += 1
                if all(count >= cap_per_class for count in class_counts.values()):
                    break
        large_test_idx = np.array(large_test_idx_recorder)
        large_test_idx = np.random.permutation(large_test_idx)

    # 计算各数据集节点数的最大/最小值
    def get_size_stats(idx_list):
        if len(idx_list) == 0:
            return 0, 0
        idx_sizes = [sizes[idx] for idx in idx_list]
        return np.max(idx_sizes), np.min(idx_sizes)

    train_max, train_min = get_size_stats(train_idx)
    val_max, val_min = get_size_stats(val_idx)
    small_test_max, small_test_min = get_size_stats(small_test_idx)
    large_test_max, large_test_min = get_size_stats(large_test_idx)

    # 打印划分结果
    print(f"==================== 文献对齐版划分结果 ====================")
    print(f"【原始划分】")
    print(f"- 训练集：{len(train_idx)} 样本 | 节点数：{train_min} ~ {train_max}")
    print(f"- 验证集：{len(val_idx)} 样本 | 节点数：{val_min} ~ {val_max}")
    print(f"- 小测试集：{len(small_test_idx)} 样本 | 节点数：{small_test_min} ~ {small_test_max}")
    print(f"- 大测试集：{len(large_test_idx)} 样本 | 节点数：{large_test_min} ~ {large_test_max}")
    print(f"【上采样后（训练集）】")
    print(
        f"- 训练集（BBBP负样本×6）：{len(train_idx) + 5 * len([i for i in train_idx if cur_data[i].y.item() == 0])} 样本")
    print(f"==========================================================")

    # 转换索引类型为int64
    train_idx = train_idx.astype(np.int64)
    val_idx = val_idx.astype(np.int64)
    small_test_idx = small_test_idx.astype(np.int64)
    large_test_idx = large_test_idx.astype(np.int64)

    return cur_data, train_idx, val_idx, small_test_idx, large_test_idx



def save_idx_files(
        train_val_test_idx_save_address,
        train_idx,
        val_idx,
        small_test_idx,
        large_test_idx
):

    os.makedirs(train_val_test_idx_save_address, exist_ok=True)

    
    pickle.dump(train_idx, open(f"{train_val_test_idx_save_address}/train_idx.pkl", "wb"))
    pickle.dump(val_idx, open(f"{train_val_test_idx_save_address}/val_idx.pkl", "wb"))
    pickle.dump(small_test_idx, open(f"{train_val_test_idx_save_address}/small_test_idx.pkl", "wb"))
    pickle.dump(large_test_idx, open(f"{train_val_test_idx_save_address}/large_test_idx.pkl", "wb"))

    print(f"索引保存完成！")
    print(f"训练集：{len(train_idx)}")
    print(f"验证集：{len(val_idx)}")
    print(f"小测试集：{len(small_test_idx)}")
    print(f"大测试集：{len(large_test_idx)}")