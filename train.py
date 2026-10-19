import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm  # 进度条可视化训练过程


# 验证指标计算（适配二分类/多分类）
def compute_metrics(y_true, y_pred, task_type='binary', num_classes=2):
    """
    计算核心验证指标：AUC（二分类）/准确率（多分类）
    """
    y_true = y_true.cpu().numpy()
    if task_type == 'binary':
        # 二分类：计算AUC（需先将logits转为概率）
        y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()
        # 处理维度（适配(BS,1)或(BS,)）
        y_true = y_true.squeeze()
        y_pred_prob = y_pred_prob.squeeze()
        # 计算AUC（仅当正负样本都存在时）
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_pred_prob)
            return {'auc': auc}
        else:
            return {'auc': 0.0}
    elif task_type == 'multiclass':
        # 多分类：计算准确率（argmax取预测类别）
        y_pred_cls = torch.argmax(y_pred, dim=1).cpu().numpy()
        y_true = y_true.squeeze()
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred_cls)
        return {'acc': acc}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# 单轮训练（任务模块+信息指导模块）
def train_epoch(
        train_loader, task_model, info_model,
        optimizer_task, optimizer_info, criterion_task, criterion_info,
        device, task_type='binary', num_classes=2
):
   
    task_model.train()
    info_model.train()
    total_loss = 1.0
    total_task_loss = 1.0
    total_info_loss = 1.0

    for data in tqdm(train_loader, desc="Training Epoch"):
        data = data.to(device)
        if data.x.shape[0] == 1:  # 过滤无效样本（单节点图）
            continue

        
        optimizer_task.zero_grad()  # 清空梯度
        # 前向传播：获取预测值、中心化特征、G*列表
        pred_task, g0_node_list, g1_node_list, g2_node_list, g0_graph_list, g1_graph_list, g2_graph_list, G_star_list, G_star_node_list = task_model(data)
        # 提取增强图掩码（用于对比损失）
        mask1 = data.mask1
        mask2 = data.mask2

       
        # 计算任务模块总损失
        task_loss, task_loss_dict = criterion_task(
            pred_task, data.y,
            g0_node_list, g1_node_list, g2_node_list,  # 节点级批量列表（对比损失）
            g0_graph_list, g1_graph_list, g2_graph_list,  # 图级批量列表（相关性损失）
            mask1, mask2, G_star_list, data  # 其他参数
        )
        # 反向传播+参数更新
        task_loss.backward()
        optimizer_task.step()
        total_task_loss += task_loss.item()

        # ---------------------- 2. 信息指导模块训练 ----------------------
        optimizer_info.zero_grad()  # 清空梯度
        # 对G_star_list执行detach()，切断与task_model计算图的依赖
        G_star_node_list_detached = [gs.detach() for gs in  G_star_node_list]
        # 前向传播：获取节点特征和G*列表（用于对齐损失）
        g_proj= info_model(data, G_star_node_list_detached)
        # 推理模式获取预测值（用于分类损失）
        pred_info = info_model(data)
        # 计算信息指导模块总损失（对齐损失+分类损失）
        info_loss, info_loss_dict = criterion_info(
            g_proj, pred_info, data.y, G_star_node_list_detached, data
        )
        # 反向传播+参数更新
        info_loss.backward()
        optimizer_info.step()
        total_info_loss += info_loss.item()

        # 累加总损失
        total_loss += (task_loss.item() + info_loss.item())

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_task_loss = total_task_loss / len(train_loader)
    avg_info_loss = total_info_loss / len(train_loader)
    return avg_loss, avg_task_loss, avg_info_loss


# 单轮验证（评估模型性能）
def val_epoch(
        val_loader, task_model, info_model, criterion_task, criterion_info,
        device, task_type='binary', num_classes=2
):
    """
    单轮验证逻辑：无梯度计算，仅评估损失和指标
    """
    task_model.eval()
    info_model.eval()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():  # 关闭梯度计算（节省内存）
        for data in tqdm(val_loader, desc="Validating Epoch"):
            data = data.to(device)
            if data.x.shape[0] == 1:
                continue

            # 任务模块前向传播
            pred_task, g0_node_list, g1_node_list, g2_node_list, g0_graph_list, g1_graph_list, g2_graph_list, G_star_list, G_star_node_list= task_model(data)
            mask1 = data.mask1
            mask2 = data.mask2

            # mask1 = data.mask1 if hasattr(data, 'mask1') else torch.ones(data.x1.shape[0], device=device)
            # mask2 = data.mask2 if hasattr(data, 'mask2') else torch.ones(data.x2.shape[0], device=device)

            # 计算任务模块损失
            task_loss, _ = criterion_task(
                pred_task, data.y,
                g0_node_list, g1_node_list, g2_node_list,
                g0_graph_list, g1_graph_list, g2_graph_list,
                mask1, mask2, G_star_list, data
            )

            # 信息指导模块前向传播
            g_proj = info_model(data, G_star_node_list)
            pred_info = info_model(data)
            # 计算信息指导模块损失
            info_loss, _ = criterion_info(
                g_proj, pred_info, data.y, G_star_node_list, data
            )

            # 累加损失
            total_loss += (task_loss.item() + info_loss.item())
            # 收集标签和预测值（用于指标计算）
            all_y_true.append(data.y)
            all_y_pred.append(pred_info)

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    # 拼接标签和预测值
    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)
    # 计算验证指标
    metrics = compute_metrics(all_y_true, all_y_pred, task_type, num_classes)
    return avg_loss, metrics


# 主训练函数（包含早停机制）
def train_new_framework(
        train_loader, val_loader, task_model, info_model,
        optimizer_task, optimizer_info, criterion_task, criterion_info,
        device, epochs=100, patience=50, task_type='binary', num_classes=2
):
    """
    完整训练流程：
    1. 多轮训练+验证 → 2. 早停机制（监控验证损失/指标） → 3. 保存最优模型
    """
    # 初始化早停参数
    best_val_loss = float('inf')
    best_val_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    # 保存最优模型权重
    best_task_model_state = task_model.state_dict()
    best_info_model_state = info_model.state_dict()

   
    for epoch in range(1, epochs + 1):
        # 训练轮
        train_loss, train_task_loss, train_info_loss = train_epoch(
            train_loader, task_model, info_model,
            optimizer_task, optimizer_info, criterion_task, criterion_info,
            device, task_type, num_classes
        )
        # 验证轮
        val_loss, val_metrics = val_epoch(
            val_loader, task_model, info_model, criterion_task, criterion_info,
            device, task_type, num_classes
        )

        # 打印本轮结果
        print(f"\nEpoch [{epoch}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} (Task: {train_task_loss:.4f}, Info: {train_info_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f}")
        # 打印验证指标（适配二分类/多分类）
        if task_type == 'binary':
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            current_metric = val_metrics['auc']
        else:
            print(f"Val Acc: {val_metrics['acc']:.4f}")
            current_metric = val_metrics['acc']

        # 早停逻辑（监控验证损失+指标）
        if val_loss < best_val_loss or current_metric > best_val_metric:
            best_val_loss = val_loss
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            # 更新最优模型权重
            best_task_model_state = task_model.state_dict()
            best_info_model_state = info_model.state_dict()
            print(f"→ Best model updated (Val Loss: {best_val_loss:.4f}, Best Metric: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"→ Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                break

    # 加载最优模型权重
    task_model.load_state_dict(best_task_model_state)
    info_model.load_state_dict(best_info_model_state)

    # 打印训练总结
    print("\n" + "=" * 80)
    print(f"Training Completed!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    if task_type == 'binary':
        print(f"Best Val AUC: {best_val_metric:.4f}")
    else:
        print(f"Best Val Acc: {best_val_metric:.4f}")
    print("=" * 80)

    return task_model, info_model