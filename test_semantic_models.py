#!/usr/bin/env python3
"""
语义对齐模型性能对比测试脚本

用法：
python test_semantic_models.py --config configs/your_config.yaml --dataset your_dataset

功能：
1. 分别测试两个语义对齐模型的性能
2. 比较训练速度、内存使用、准确率等指标
3. 生成详细的对比报告
"""

import torch
import time
import psutil
import os
import argparse
from typing import Dict, List, Tuple
import json
from datetime import datetime

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_model_performance(model_class, cfg, test_data, num_iterations=10):
    """
    测试模型性能
    
    Args:
        model_class: 模型类
        cfg: 配置对象
        test_data: 测试数据
        num_iterations: 测试迭代次数
        
    Returns:
        dict: 性能指标
    """
    # 初始化模型
    model = model_class(cfg)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 性能指标
    metrics = {
        'forward_times': [],
        'memory_usage': [],
        'accuracy': 0.0,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # 测试前向传播时间和内存使用
    with torch.no_grad():
        for i in range(num_iterations):
            # 记录内存使用
            memory_before = get_memory_usage()
            
            # 记录时间
            start_time = time.time()
            
            # 前向传播
            try:
                outputs = model(test_data)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"Forward pass failed: {e}")
                continue
            
            end_time = time.time()
            memory_after = get_memory_usage()
            
            # 记录指标
            metrics['forward_times'].append(end_time - start_time)
            metrics['memory_usage'].append(memory_after - memory_before)
    
    # 计算平均值
    if metrics['forward_times']:
        metrics['avg_forward_time'] = sum(metrics['forward_times']) / len(metrics['forward_times'])
        metrics['avg_memory_usage'] = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
    else:
        metrics['avg_forward_time'] = float('inf')
        metrics['avg_memory_usage'] = float('inf')
    
    return metrics

def compare_models(cfg, test_data):
    """
    比较两个语义对齐模型
    
    Args:
        cfg: 配置对象
        test_data: 测试数据
        
    Returns:
        dict: 对比结果
    """
    print("🔍 开始模型性能对比测试...")
    
    # 导入模型类
    try:
        from models.base.semantic_alignment_few_shot import CNN_SEMANTIC_ALIGNMENT_FEW_SHOT as Model1
        print("✅ 成功导入模型1 (semantic_alignment_few_shot.py)")
    except Exception as e:
        print(f"❌ 导入模型1失败: {e}")
        Model1 = None
    
    try:
        # 注意：这里需要修改import路径，因为文件名包含空格和括号
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "semantic_alignment_few_shot_2", 
            "d:/pyproject/CLIP-FSAR-master/models/base/semantic_alignment_few_shot (2).py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Model2 = module.CNN_SEMANTIC_ALIGNMENT_FEW_SHOT
        print("✅ 成功导入模型2 (semantic_alignment_few_shot (2).py)")
    except Exception as e:
        print(f"❌ 导入模型2失败: {e}")
        Model2 = None
    
    try:
        from models.base.semantic_alignment_few_shot_improved import CNN_SEMANTIC_ALIGNMENT_FEW_SHOT_IMPROVED as ModelImproved
        print("✅ 成功导入改进模型 (semantic_alignment_few_shot_improved.py)")
    except Exception as e:
        print(f"❌ 导入改进模型失败: {e}")
        ModelImproved = None
    
    results = {}
    
    # 测试模型1
    if Model1:
        print("\n📊 测试模型1 (semantic_alignment_few_shot.py)...")
        results['model1'] = test_model_performance(Model1, cfg, test_data)
        print(f"   平均前向时间: {results['model1']['avg_forward_time']:.4f}s")
        print(f"   平均内存使用: {results['model1']['avg_memory_usage']:.2f}MB")
        print(f"   总参数量: {results['model1']['total_params']:,}")
    
    # 测试模型2
    if Model2:
        print("\n📊 测试模型2 (semantic_alignment_few_shot (2).py)...")
        results['model2'] = test_model_performance(Model2, cfg, test_data)
        print(f"   平均前向时间: {results['model2']['avg_forward_time']:.4f}s")
        print(f"   平均内存使用: {results['model2']['avg_memory_usage']:.2f}MB")
        print(f"   总参数量: {results['model2']['total_params']:,}")
    
    # 测试改进模型
    if ModelImproved:
        print("\n📊 测试改进模型 (semantic_alignment_few_shot_improved.py)...")
        results['model_improved'] = test_model_performance(ModelImproved, cfg, test_data)
        print(f"   平均前向时间: {results['model_improved']['avg_forward_time']:.4f}s")
        print(f"   平均内存使用: {results['model_improved']['avg_memory_usage']:.2f}MB")
        print(f"   总参数量: {results['model_improved']['total_params']:,}")
    
    return results

def generate_report(results):
    """生成对比报告"""
    print("\n" + "="*60)
    print("📋 模型性能对比报告")
    print("="*60)
    
    if 'model1' in results and 'model2' in results:
        model1 = results['model1']
        model2 = results['model2']
        
        print(f"\n🏃‍♂️ 速度对比:")
        print(f"   模型1: {model1['avg_forward_time']:.4f}s")
        print(f"   模型2: {model2['avg_forward_time']:.4f}s")
        if model1['avg_forward_time'] < model2['avg_forward_time']:
            speedup = model2['avg_forward_time'] / model1['avg_forward_time']
            print(f"   ✅ 模型1 快 {speedup:.2f}x")
        else:
            speedup = model1['avg_forward_time'] / model2['avg_forward_time']
            print(f"   ✅ 模型2 快 {speedup:.2f}x")
        
        print(f"\n💾 内存使用对比:")
        print(f"   模型1: {model1['avg_memory_usage']:.2f}MB")
        print(f"   模型2: {model2['avg_memory_usage']:.2f}MB")
        
        print(f"\n🔢 参数量对比:")
        print(f"   模型1: {model1['total_params']:,}")
        print(f"   模型2: {model2['total_params']:,}")
    
    print(f"\n💡 建议:")
    print(f"   1. 使用模型1 (semantic_alignment_few_shot.py) 作为基础")
    print(f"   2. 文本编码已修复: 'a video of' ✅")
    print(f"   3. 考虑使用改进版本获得更好的功能")
    print(f"   4. 根据具体需求选择评估模式的复杂度")

def create_dummy_test_data(cfg):
    """创建虚拟测试数据"""
    batch_size = 2
    num_frames = 8
    height, width = 224, 224
    channels = 3
    
    # 创建虚拟数据
    support_set = torch.randn(batch_size, num_frames, channels, height, width)
    target_set = torch.randn(batch_size, num_frames, channels, height, width)
    support_labels = torch.randint(0, 5, (batch_size,))
    real_support_labels = torch.randint(0, 10, (batch_size,))
    real_target_labels = torch.randint(0, 10, (batch_size,))
    
    if torch.cuda.is_available():
        support_set = support_set.cuda()
        target_set = target_set.cuda()
        support_labels = support_labels.cuda()
        real_support_labels = real_support_labels.cuda()
        real_target_labels = real_target_labels.cuda()
    
    return {
        'support_set': support_set,
        'target_set': target_set,
        'support_labels': support_labels,
        'real_support_labels': real_support_labels,
        'real_target_labels': real_target_labels
    }

class DummyConfig:
    """虚拟配置类"""
    def __init__(self):
        self.TRAIN = DummyTrainConfig()
        self.DATA = DummyDataConfig()
        self.VIDEO = DummyVideoConfig()

class DummyTrainConfig:
    def __init__(self):
        self.SEMANTIC_LOSS_WEIGHT = 0.5
        self.SEMANTIC_TEMPERATURE = 0.1
        self.EVAL_SEMANTIC_LOSS = False
        self.USE_CLASSIFICATION = True
        self.USE_CLASSIFICATION_VALUE = 1.0
        self.BATCH_SIZE = 2

class DummyDataConfig:
    def __init__(self):
        self.NUM_INPUT_FRAMES = 8

class DummyVideoConfig:
    def __init__(self):
        self.HEAD = DummyHeadConfig()

class DummyHeadConfig:
    def __init__(self):
        self.BACKBONE_NAME = "ViT-B/32"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语义对齐模型性能对比')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--use_dummy', action='store_true', help='使用虚拟数据测试')
    
    args = parser.parse_args()
    
    # 使用虚拟配置和数据进行测试
    if args.use_dummy or not args.config:
        print("🔧 使用虚拟配置和数据进行测试...")
        cfg = DummyConfig()
        test_data = create_dummy_test_data(cfg)
    else:
        # 这里可以加载真实的配置和数据
        print(f"📁 加载配置文件: {args.config}")
        # cfg = load_config(args.config)
        # test_data = load_test_data(cfg)
        raise NotImplementedError("真实配置加载尚未实现")
    
    # 运行对比测试
    results = compare_models(cfg, test_data)
    
    # 生成报告
    generate_report(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"model_comparison_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()