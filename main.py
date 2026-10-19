import argparse
import torch
import os
import numpy as np
import random
import pickle
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset
from graphsst2_dataset import get_dataset
import datapreprocessing_new as dp
import model_new as models
import criteria_new as criteria
import train_util_new as train_util

# 设置随机种子（保证实验可复现）
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# 命令行参数解析
parser = argparse.ArgumentParser()

bbbp 
parser.add_argument('--dataset', default="bbbp", choices=["bbbp", "PROTEINS", "nci1", "graphsst2"], help='datasets')
parser.add_argument('--dataset_root', default="", help='datasets root')
parser.add_argument('--train_val_test_idx_save_root', default="",
                    help='index save root')
parser.add_argument('--model_save_path', default=".",
                    help='Path to save trained models')


parser.add_argument('--batch_size', default=32, type=int, help='batch size')  
parser.add_argument('--gpu_id', default="0", help='GPU id')
parser.add_argument('--lr', default=1, type=float, help='learning rate')  
parser.add_argument('--epochs', default=100, type=int, help='total epochs')  
parser.add_argument('--patience', default=80, type=int, help='patience for early stopping')
parser.add_argument('--hidden_dim', default=256, type=int, help='GNN hidden dim')  
parser.add_argument('--task_dim', default=256, type=int, help='task space dim')  
parser.add_argument('--tau_cl', default=1.0, type=float, help='contrastive loss temperature')
parser.add_argument('--alpha', default=1.0, type=float, help='node CL loss weight')
parser.add_argument('--beta', default=1.0, type=float, help='edge consistency loss weight')
parser.add_argument('--gamma', default=1.0, type=float, help='task correlation loss weight')
parser.add_argument('--delta', default=1.0, type=float, help='CE loss weight')
parser.add_argument('--align_alpha', default=1.0, type=float, help='alignment loss weight')
parser.add_argument('--align_beta', default=1.0, type=float, help='info CE loss weight')
parser.add_argument('--task_type', default=None, choices=['binary', 'multiclass'],
                    help='Task type (auto-detected if None)')
parser.add_argument('--num_classes', default=None, type=int,
                    help='Number of classes (auto-detected if None)')
args = parser.parse_args()






# 加载数据集
print(f"Loading dataset: {args.dataset}")
cur_data = PygGraphPropPredDataset(name="ogbg-molbbbp", pre_transform=None, root=args.dataset_root)
dataset_root = args.dataset_root + "_bbbp"


# 自动识别任务配置
task_type, num_classes = auto_detect_task_info(args.dataset, cur_data)
task_type = args.task_type if args.task_type is not None else task_type
num_classes = args.num_classes if args.num_classes is not None else num_classes

# 数据划分（若已存在索引文件则直接加载，否则重新划分）
train_val_test_idx_save_address = os.path.join(args.train_val_test_idx_save_root, args.dataset)
if not os.path.exists(train_val_test_idx_save_address):
    os.makedirs(train_val_test_idx_save_address)
    # 调用数据划分函数（已包含节点数过滤逻辑）
    cur_data_filtered, train_idx, val_idx, small_test_idx, large_test_idx = dp.quick_test_data_prep(cur_data, task_type, num_classes)
    # dp.save_idx_files(train_val_test_idx_save_address, cur_data_filtered,train_idx, val_idx, small_test_idx, large_test_idx)
    dp.save_idx_files(train_val_test_idx_save_address, train_idx, val_idx, small_test_idx, large_test_idx)
else:
    # 加载已保存的索引
    cur_data_filtered = pickle.load(open(f"{train_val_test_idx_save_address}/cur_data_filtered.pkl", "rb"))
    train_idx = pickle.load(open(f"{train_val_test_idx_save_address}/train_idx.pkl", "rb"))
    val_idx = pickle.load(open(f"{train_val_test_idx_save_address}/val_idx.pkl", "rb"))
  

# 转换索引类型并构建数据集
train_idx = train_idx.astype(np.int64)
val_idx = val_idx.astype(np.int64)
train_dataset = [cur_data_filtered[idx] for idx in train_idx]   
val_dataset = [cur_data_filtered[idx] for idx in val_idx]     


# 二分类数据集执行上采样（多分类不执行）
if args.dataset != "graphsst2":
    print("Upsampling training dataset...")
    #train_dataset = dp.UpsampledDataset(cur_data_filtered,train_dataset, dataset_root, args.dataset)
    train_dataset = dp.UpsampledDataset(
        cur_dataset=train_dataset, 
        dataset_root=dataset_root,  
        dataset_name=args.dataset,  
        task_type=task_type,  #
        num_classes=num_classes,  
        min_node_num=8  
    )


# 生成增强图数据集
print("Generating augmented datasets...")
train_dataset_aug = dp.AugmentedDatasetNew(
    cur_dataset=train_dataset,          
    dataset_root=dataset_root,          
    split_name="train",                 
    task_type=task_type,                
    num_classes=num_classes,            
    sample_ratios=[0.15, 0.3],           
    device=device,                      
    pre_filter=None,                    
    pre_transform=None                 
)

val_dataset_aug = dp.AugmentedDatasetNew(
    cur_dataset=val_dataset,            
    dataset_root=dataset_root,          
    split_name="val",                   
    task_type=task_type,                
    num_classes=num_classes,            
    sample_ratios=[0.15, 0.3],         
    device=device,                      
    pre_filter=None,
    pre_transform=None
)






# 构建数据加载器
train_loader = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True,follow_batch=['x1', 'x2'])
val_loader = DataLoader(val_dataset_aug, batch_size=args.batch_size, shuffle=False,follow_batch=['x1', 'x2'])

# 初始化模型
input_dim = train_dataset[0].x.shape[-1] if train_dataset[0].x is not None else 1
task_model = models.TaskInfoExtractor(
    input_dim=input_dim,
    hidden_dim=args.hidden_dim,
    task_dim=args.task_dim,
    num_classes=num_classes
).to(device)
info_model = models.InfoGuideModel(
    input_dim=input_dim,
    hidden_dim=args.hidden_dim,
    task_dim=args.task_dim,
    num_classes=num_classes
).to(device)




optimizer_task = torch.optim.AdamW(
    task_model.parameters(),
    lr=args.lr,
    weight_decay=1
)

optimizer_info = torch.optim.AdamW(
    info_model.parameters(),
    lr=args.lr,
    weight_decay=1
)

# 初始化损失函数
criterion_task = criteria.NewFrameworkCriterion(
    tau_cl=args.tau_cl,
    alpha=args.alpha,
    beta=args.beta,
    gamma=args.gamma,
    delta=args.delta,
    task_type=task_type,
    num_classes=num_classes
)
criterion_info = criteria.InfoGuideCriterion(
    alpha=args.align_alpha,
    beta=args.align_beta,
    task_type=task_type,
    num_classes=num_classes
)

# 开始训练
print("Starting training...")
trained_task_model, trained_info_model = train_util.train_new_framework(
    train_loader, val_loader,
    task_model, info_model, optimizer_task, optimizer_info,
    criterion_task, criterion_info, device,
    epochs=args.epochs, patience=args.patience,
    task_type=task_type, num_classes=num_classes
)

# 保存训练好的模型
os.makedirs(args.model_save_path, exist_ok=True)
info_model_path = os.path.join(args.model_save_path, f"{args.dataset}_info_model.pth")
torch.save({
    'model_state_dict': trained_info_model.state_dict(),
    'input_dim': input_dim,
    'hidden_dim': args.hidden_dim,
    'task_dim': args.task_dim,
    'num_classes': num_classes,
    'task_type': task_type
}, info_model_path)
task_model_path = os.path.join(args.model_save_path, f"{args.dataset}_task_model.pth")
torch.save({
    'model_state_dict': trained_task_model.state_dict(),
    'input_dim': input_dim,
    'hidden_dim': args.hidden_dim,
    'task_dim': args.task_dim,
    'num_classes': num_classes,
    'task_type': task_type
}, task_model_path)

print(f"\nTraining completed!")
print(f"InfoGuideModel saved to: {info_model_path}")
print(f"TaskInfoExtractor saved to: {task_model_path}")
