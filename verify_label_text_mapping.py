#!/usr/bin/env python3
"""
验证视频标签与文本特征对应关系的脚本
检查在语义对齐损失计算中，视频的类别标签是否正确对应到相应的文本阶段特征
"""

import torch
import numpy as np
from models.base.few_shot import load, tokenize
import re
from typing import Dict, List

def create_simple_config():
    """创建简单的配置对象"""
    class SimpleConfig:
        def __init__(self):
            # 模拟HMDB51数据集的一些类别
            self.TRAIN = type('Train', (), {
                'CLASS_NAME': [
                    'brush hair',
                    'cartwheel', 
                    'catch',
                    'chew',
                    'clap',
                    'climb',
                    'climb stairs',
                    'dive',
                    'draw sword',
                    'dribble'
                ]
            })()
            
            self.TEST = type('Test', (), {
                'CLASS_NAME': self.TRAIN.CLASS_NAME
            })()
            
            self.VIDEO = type('Video', (), {
                'HEAD': type('Head', (), {
                    'BACKBONE_NAME': 'RN50'
                })()
            })()
    
    return SimpleConfig()

def parse_semantic_stages(class_names: List[str]) -> Dict[str, List[str]]:
    """解析类别名称的语义阶段（模拟原始代码的逻辑）"""
    semantic_stages = {}
    
    for class_name in class_names:
        # 使用逗号和"then"分割语义阶段
        stages = re.split(r'[,，]\s*|then\s+', class_name.lower())
        stages = [stage.strip() for stage in stages if stage.strip()]
        
        if len(stages) > 1:
            semantic_stages[class_name] = stages
        else:
            # 如果没有明确的分割，保持原始名称
            semantic_stages[class_name] = [class_name]
            
    return semantic_stages

def create_stage_text_features(semantic_stages: Dict[str, List[str]], clip_model) -> Dict[str, torch.Tensor]:
    """为每个类别的语义阶段创建文本特征（模拟原始代码的逻辑）"""
    stage_text_features = {}
    
    # 使用torch.no_grad()避免梯度计算问题
    with torch.no_grad():
        for class_name, stages in semantic_stages.items():
            stage_features = []
            for stage in stages:
                # 使用CLIP编码每个阶段的文本
                stage_text = f"a photo of {stage}"
                # 使用完整的CLIP模型进行文本编码
                text_tokens = tokenize([stage_text]).cuda()
                stage_feature = clip_model.encode_text(text_tokens)
                stage_features.append(stage_feature)
            
            # 堆叠所有阶段特征 [num_stages, feature_dim]
            stage_text_features[class_name] = torch.cat(stage_features, dim=0)
        
    return stage_text_features

def verify_label_text_mapping():
    """验证标签与文本特征的对应关系"""
    print("=" * 80)
    print("验证视频标签与文本特征对应关系")
    print("=" * 80)
    
    # 创建配置
    cfg = create_simple_config()
    
    # 加载CLIP模型
    print("正在加载CLIP RN50模型...")
    clip_model, _ = load('RN50', device="cuda", cfg=cfg, jit=False)
    print("✓ CLIP RN50模型加载成功")
    
    # 获取类别名称列表（模拟class_real_train）
    class_real_train = cfg.TRAIN.CLASS_NAME
    print(f"\n类别名称列表 (class_real_train):")
    for i, class_name in enumerate(class_real_train):
        print(f"  索引 {i}: '{class_name}'")
    
    # 解析语义阶段
    print(f"\n解析语义阶段:")
    semantic_stages = parse_semantic_stages(class_real_train)
    for class_name, stages in semantic_stages.items():
        print(f"  '{class_name}' -> {stages}")
    
    # 创建文本特征
    print(f"\n创建文本特征...")
    stage_text_features = create_stage_text_features(semantic_stages, clip_model)
    
    # 验证对应关系
    print(f"\n验证标签与文本特征的对应关系:")
    print("-" * 60)
    
    # 模拟一些视频标签
    test_labels = torch.tensor([0, 2, 5, 7, 9])  # 一些测试标签
    
    for i, class_idx in enumerate(test_labels):
        class_idx_val = class_idx.item()
        
        print(f"\n测试样本 {i+1}:")
        print(f"  视频标签索引: {class_idx_val}")
        
        # 检查索引是否在有效范围内
        if class_idx_val < len(class_real_train):
            class_name = class_real_train[class_idx_val]
            print(f"  对应类别名称: '{class_name}'")
            
            # 检查是否有对应的文本特征
            if class_name in stage_text_features:
                text_features = stage_text_features[class_name]
                num_stages = text_features.shape[0]
                feature_dim = text_features.shape[1]
                print(f"  文本特征维度: [{num_stages}, {feature_dim}]")
                print(f"  语义阶段: {semantic_stages[class_name]}")
                print(f"  ✓ 标签与文本特征匹配正确")
            else:
                print(f"  ✗ 错误：找不到类别 '{class_name}' 的文本特征")
        else:
            print(f"  ✗ 错误：标签索引 {class_idx_val} 超出范围 (最大: {len(class_real_train)-1})")
    
    # 额外验证：检查所有类别都有对应的文本特征
    print(f"\n完整性检查:")
    print("-" * 60)
    missing_features = []
    for i, class_name in enumerate(class_real_train):
        if class_name not in stage_text_features:
            missing_features.append((i, class_name))
    
    if missing_features:
        print(f"✗ 发现缺失的文本特征:")
        for idx, name in missing_features:
            print(f"  索引 {idx}: '{name}'")
    else:
        print(f"✓ 所有类别都有对应的文本特征")
    
    # 验证在_compute_semantic_alignment_loss中的逻辑
    print(f"\n模拟 _compute_semantic_alignment_loss 中的逻辑:")
    print("-" * 60)
    
    batch_size = len(test_labels)
    for i in range(batch_size):
        class_idx = test_labels[i].item()
        print(f"\nBatch样本 {i}:")
        print(f"  class_labels[{i}] = {class_idx}")
        
        # 确保class_idx在有效范围内
        if class_idx < len(class_real_train):
            class_name = class_real_train[class_idx]
            print(f"  class_real_train[{class_idx}] = '{class_name}'")
            
            # 获取该类别的语义阶段文本特征
            if class_name in stage_text_features:
                stage_text_features_for_class = stage_text_features[class_name]
                print(f"  stage_text_features['{class_name}'].shape = {stage_text_features_for_class.shape}")
                print(f"  ✓ 成功获取到对应的文本特征")
            else:
                print(f"  ✗ 错误：无法获取类别 '{class_name}' 的文本特征")
        else:
            print(f"  ✗ 错误：class_idx {class_idx} 超出范围")
    
    print(f"\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == "__main__":
    verify_label_text_mapping()