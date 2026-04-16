#!/usr/bin/env python3
"""
验证修改后的语义对齐模型：每个视频只与其对应标签计算相似度
"""

import torch
import torch.nn.functional as F
import numpy as np

def cos_sim(x, y, epsilon=1e-8):
    """计算余弦相似度"""
    # 归一化
    x_norm = F.normalize(x, p=2, dim=-1, eps=epsilon)
    y_norm = F.normalize(y, p=2, dim=-1, eps=epsilon)
    
    # 计算余弦相似度
    return torch.matmul(x_norm, y_norm.transpose(-2, -1))

def verify_label_only_similarity():
    """验证每个视频只与其对应标签计算相似度的逻辑"""
    print("=" * 60)
    print("验证修改后的语义对齐模型：每个视频只与其对应标签计算相似度")
    print("=" * 60)
    
    # 模拟参数
    batch_size = 4
    feature_dim = 1024
    num_classes = 51  # HMDB51数据集
    temperature = 0.1
    
    # 模拟数据
    torch.manual_seed(42)
    feature_classification = torch.randn(batch_size, feature_dim)  # 视频特征
    text_features_train = torch.randn(num_classes, feature_dim)   # 所有类别的文本特征
    labels = torch.tensor([5, 12, 23, 8])  # 模拟标签
    
    print(f"批次大小: {batch_size}")
    print(f"特征维度: {feature_dim}")
    print(f"类别数量: {num_classes}")
    print(f"标签: {labels.tolist()}")
    print()
    
    # 原始方法：与所有类别计算相似度
    print("1. 原始方法：与所有类别计算相似度")
    all_similarities = cos_sim(feature_classification, text_features_train) / temperature
    print(f"   相似度矩阵形状: {all_similarities.shape}")  # [batch_size, num_classes]
    print(f"   相似度范围: [{all_similarities.min().item():.4f}, {all_similarities.max().item():.4f}]")
    
    # 提取正确标签的相似度
    correct_similarities_original = []
    for i in range(batch_size):
        correct_similarities_original.append(all_similarities[i, labels[i]])
    correct_similarities_original = torch.stack(correct_similarities_original)
    print(f"   正确标签相似度: {correct_similarities_original.detach().numpy()}")
    print()
    
    # 修改后的方法：只与对应标签计算相似度
    print("2. 修改后的方法：只与对应标签计算相似度")
    label_similarities = []
    for i in range(batch_size):
        class_idx = labels[i].item()
        # 获取对应标签的文本特征
        target_text_feature = text_features_train[class_idx:class_idx+1]  # [1, feature_dim]
        # 计算当前样本与其标签的相似度
        sample_similarity = cos_sim(feature_classification[i:i+1], target_text_feature) / temperature  # [1, 1]
        label_similarities.append(sample_similarity.squeeze())  # 标量
    
    # 堆叠所有相似度为一个张量 [batch_size]
    text_similarity = torch.stack(label_similarities)
    print(f"   相似度向量形状: {text_similarity.shape}")  # [batch_size]
    print(f"   相似度范围: [{text_similarity.min().item():.4f}, {text_similarity.max().item():.4f}]")
    print(f"   标签相似度: {text_similarity.detach().numpy()}")
    print()
    
    # 验证两种方法的结果是否一致
    print("3. 验证两种方法的结果一致性")
    difference = torch.abs(correct_similarities_original - text_similarity)
    print(f"   差异: {difference.detach().numpy()}")
    print(f"   最大差异: {difference.max().item():.8f}")
    print(f"   结果一致: {torch.allclose(correct_similarities_original, text_similarity, atol=1e-6)}")
    print()
    
    # 损失计算对比
    print("4. 损失计算对比")
    
    # 原始交叉熵损失
    ce_loss = F.cross_entropy(all_similarities, labels)
    print(f"   原始交叉熵损失: {ce_loss.item():.6f}")
    
    # 修改后的负相似度损失
    neg_sim_loss = -text_similarity.mean()
    print(f"   修改后负相似度损失: {neg_sim_loss.item():.6f}")
    print()
    
    # 分析损失的含义
    print("5. 损失含义分析")
    print("   原始方法：")
    print("   - 使用交叉熵损失，鼓励正确类别的相似度高，其他类别的相似度低")
    print("   - 损失考虑了与所有类别的相对关系")
    print()
    print("   修改后方法：")
    print("   - 使用负相似度损失，直接最大化与正确标签的相似度")
    print("   - 损失只关注与正确标签的相似度，不考虑其他类别")
    print("   - 相似度越高，损失越小")
    print()
    
    # 梯度分析
    print("6. 梯度流分析")
    feature_classification.requires_grad_(True)
    
    # 原始方法的梯度
    ce_loss_grad = F.cross_entropy(cos_sim(feature_classification, text_features_train) / temperature, labels)
    ce_loss_grad.backward(retain_graph=True)
    ce_grad_norm = torch.norm(feature_classification.grad).item()
    feature_classification.grad.zero_()
    
    # 修改后方法的梯度
    new_similarities = []
    for i in range(batch_size):
        class_idx = labels[i].item()
        target_text_feature = text_features_train[class_idx:class_idx+1]
        sample_similarity = cos_sim(feature_classification[i:i+1], target_text_feature) / temperature
        new_similarities.append(sample_similarity.squeeze())
    
    new_sim_loss = -torch.stack(new_similarities).mean()
    new_sim_loss.backward()
    new_grad_norm = torch.norm(feature_classification.grad).item()
    
    print(f"   原始方法梯度范数: {ce_grad_norm:.6f}")
    print(f"   修改后方法梯度范数: {new_grad_norm:.6f}")
    print()
    
    print("验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    verify_label_only_similarity()