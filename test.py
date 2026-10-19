import argparse
import os
import torch
import pickle
import numpy as np
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from graphsst2_dataset import get_dataset


# 导入自定义模型
import model_new as models

# 独立实现AUC计算（避免依赖训练模块）
def eval_rocauc(y_true, y_pred, task_type='binary', num_classes=2):
    if task_type == 'binary':
        rocauc_list = []
        y_true = y_true.squeeze() if y_true.ndim > 1 else y_true
        y_pred = y_pred.squeeze() if y_pred.ndim > 1 else y_pred
        # 仅当同时存在正负样本时计算AUC
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true, y_pred))
       
            return 0.5
        return sum(rocauc_list) / len(rocauc_list)
    elif task_type == 'multiclass':
        # 多分类AUC：将标签二值化后计算每个类的AUC
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))
        rocauc_list = []
        for i in range(num_classes):
            if np.sum(y_true_binarized[:, i] == 1) > 0 and np.sum(y_true_binarized[:, i] == 0) > 0:
                rocauc_list.append(roc_auc_score(y_true_binarized[:, i], y_pred[:, i]))
        
            return 0.5
        return sum(rocauc_list) / len(rocauc_list)
   



#测试阶段AUC计算
def compute_auc(y_true, y_pred, task_type='binary', num_classes=2):
    """
    完全对齐train_util_new.txt中的compute_metrics逻辑，仅计算AUC指标
    """
    y_true = y_true.squeeze()  # 处理维度（适配(BS,1)或(BS,)）
    y_pred = y_pred.squeeze()

    if task_type == 'binary':
        # 二分类：先将logits转为概率（与训练代码一致）
        y_pred_prob = torch.sigmoid(torch.tensor(y_pred)).numpy()
        # 仅当同时存在正负样本时计算AUC
        if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
            auc = roc_auc_score(y_true, y_pred_prob)
        else:
            print("Warning: No positive/negative samples in binary task, return AUC=0.5")
            auc = 0.5
        return auc
    elif task_type == 'multiclass':
        # 多分类：标签二值化，计算每个类的AUC后平均（与训练代码一致）
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))
        rocauc_list = []
        for i in range(num_classes):
            if np.sum(y_true_binarized[:, i] == 1) > 0 and np.sum(y_true_binarized[:, i] == 0) > 0:
                rocauc_list.append(roc_auc_score(y_true_binarized[:, i], y_pred[:, i]))
        if len(rocauc_list) == 0:
            print("Warning: No positive/negative samples in multiclass task, return AUC=0.5")
            return 0.5
        return sum(rocauc_list) / len(rocauc_list)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")



def test_step(info_model, loader, device, task_type='binary', num_classes=2):
    info_model.eval()  # 推理模式（关闭Dropout等）
    y_true = []  # 真实标签
    y_pred = []  # 模型输出logits（未转概率）
    with torch.no_grad():  # 关闭梯度计算（节省内存）
        for data in loader:
            data = data.to(device)
            # 过滤空图/单节点图，强制转换x为Float型
            if data.x is None or data.x.shape[0] <= 1:
                print(f"Skip invalid sample (nodes: {data.x.shape[0] if data.x is not None else 0})")
                continue
            data.x = data.x.float()
            pred_info = info_model(data)  # 输出logits（未转概率，与训练一致）
            # 收集标签和logits（保留原始logits，后续统一计算）
            y_true.append(data.y.view(pred_info.shape).detach().cpu())
            y_pred.append(pred_info.detach().cpu())
    # 处理测试集为空的情况
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: No valid samples in test loader")
        return {
            'rocauc': 0.0, 'f1': 0.0, 'acc': 0.0, 'auc': 0.0,  # 新增auc字段
            'y_true': np.array([]), 'y_pred_prob': np.array([]), 'y_pred_hard': np.array([])
        }
    # 拼接结果并转换为numpy
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred_logits = torch.cat(y_pred, dim=0).numpy()  # 保留logits用于AUC计算
    # 预测后处理（适配二分类/多分类）
    if task_type == 'binary':
        y_pred_prob = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()  # 转概率用于F1/Acc
        y_pred_hard = (y_pred_prob > 0.5).astype(int)  # 硬预测
    elif task_type == 'multiclass':
        y_pred_prob = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()  # 转概率
        y_pred_hard = y_pred_prob.argmax(axis=1)  # 硬预测
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    # 计算核心指标（新增AUC计算，调用对齐训练逻辑的函数）
    auc = compute_auc(y_true, y_pred_logits, task_type, num_classes)  # 用logits计算AUC
    rocauc = auc  # 与原代码字段兼容（实际与auc完全一致）
    f1 = f1_score(y_true, y_pred_hard, average='macro' if task_type == 'multiclass' else 'binary')
    acc = accuracy_score(y_true, y_pred_hard)
    return {
        'rocauc': rocauc,
        'auc': auc,  # 新增独立auc字段，更直观
        'f1': f1,
        'acc': acc,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'y_pred_hard': y_pred_hard
    }

# 命令行参数解析
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default="bbbp", choices=["bbbp"])
parser.add_argument('--dataset_root', default="")
parser.add_argument('--train_val_test_idx_save_root', default="")
parser.add_argument('--info_model_path', default="")



parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--gpu_id', default="0")
parser.add_argument('--task_type', default=None, choices=['binary', 'multiclass'])
parser.add_argument('--num_classes', default=None, type=int)
args = parser.parse_args()



# 加载原始测试数据集（无增强）
print(f"Loading original test dataset: {args.dataset}")
cur_data = PygGraphPropPredDataset(name="ogbg-molbbbp", root=args.dataset_root)


# 自动识别任务配置
task_type, num_classes = auto_detect_task_info(args.dataset, cur_data)
task_type = args.task_type if args.task_type is not None else task_type
num_classes = args.num_classes if args.num_classes is not None else num_classes
print(f"Final task config - Type: {task_type}, Classes: {num_classes}")

# 加载测试集索引（已过滤节点数）
train_val_test_idx_save_address = os.path.join(args.train_val_test_idx_save_root, args.dataset)

required_idx_files = ['small_test_idx.pkl', 'large_test_idx.pkl']
for idx_file in required_idx_files:
    idx_path = os.path.join(train_val_test_idx_save_address, idx_file)
   
with open(f"{train_val_test_idx_save_address}/small_test_idx.pkl", "rb") as f:
    small_test_idx = pickle.load(f)
with open(f"{train_val_test_idx_save_address}/large_test_idx.pkl", "rb") as f:
    large_test_idx = pickle.load(f)

def validate_idx(idx_list, dataset_len, idx_name):
    if len(idx_list) == 0:
        raise ValueError(f"{idx_name} is empty")
    max_idx = max(idx_list) if len(idx_list) > 0 else -1
    if max_idx >= dataset_len:
        raise IndexError(f"{idx_name} max index ({max_idx}) >= dataset length ({dataset_len})")
    return idx_list

dataset_len = len(cur_data)
small_test_idx = validate_idx(small_test_idx, dataset_len, "small_test_idx")
large_test_idx = validate_idx(large_test_idx, dataset_len, "large_test_idx")

# 构建测试数据集和加载器
small_test_dataset = cur_data[small_test_idx]
large_test_dataset = cur_data[large_test_idx]
print(f"Small test set size: {len(small_test_dataset)}")
print(f"Large test set size: {len(large_test_dataset)}")
test_loader_small = DataLoader(small_test_dataset, batch_size=args.batch_size, shuffle=False)
test_loader_large = DataLoader(large_test_dataset, batch_size=args.batch_size, shuffle=False)

# 加载训练好的InfoGuideModel
print(f"Loading trained InfoGuideModel from: {args.info_model_path}")
if not os.path.exists(args.info_model_path):
    raise FileNotFoundError(f"Model file not found: {args.info_model_path}")
checkpoint = torch.load(args.info_model_path, map_location=device)
# 修正7：校验checkpoint关键字段，避免KeyError
required_checkpoint_keys = ['model_state_dict', 'input_dim', 'hidden_dim', 'task_dim', 'num_classes', 'task_type']
for key in required_checkpoint_keys:
    if key not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: {key}")
input_dim = checkpoint['input_dim']
hidden_dim = checkpoint['hidden_dim']
task_dim = checkpoint['task_dim']
num_classes = checkpoint['num_classes']
task_type = checkpoint['task_type']

# 初始化模型并加载权重
info_model = models.InfoGuideModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    task_dim=task_dim,
    num_classes=num_classes
).to(device)
# 修正8：处理多GPU训练的权重（如DataParallel保存的权重）
state_dict = checkpoint['model_state_dict']
# 若权重键名含"module."，移除前缀（适配单GPU/CPU加载）
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
info_model.load_state_dict(state_dict)
print("Model loaded successfully!")

# 执行测试
print("\nStarting independent test (only original test data, no augmentation)...")
small_test_res = test_step(info_model, test_loader_small, device, task_type, num_classes)
large_test_res = test_step(info_model, test_loader_large, device, task_type, num_classes)

# 修正9：完整打印测试结果（包含所有指标和数据集）
print("\n=== Final Test Results (InfoGuideModel) ===")
print(f"[Small Test Set] (Size: {len(small_test_dataset)})")
print(f"  AUC: {small_test_res['auc']:.4f}")  # 新增AUC输出
print(f"  F1 Score: {small_test_res['f1']:.4f}")
print(f"  Accuracy: {small_test_res['acc']:.4f}")


print(f"\n[Large Test Set] (Size: {len(large_test_dataset)})")
print(f"  AUC: {large_test_res['auc']:.4f}")  # 新增AUC输出
print(f"  F1 Score: {large_test_res['f1']:.4f}")
print(f"  Accuracy: {large_test_res['acc']:.4f}")