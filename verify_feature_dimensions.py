#!/usr/bin/env python3
"""
验证视频和文本编码特征维度是否相同
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def cos_sim(x, y, epsilon=0.01):
    """
    计算两个张量最后一个维度之间的余弦相似度
    这是从semantic_alignment_supervised_supervised.py第131行使用的相同函数
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def verify_feature_dimensions():
    """验证视频和文本特征维度"""
    print("=" * 60)
    print("验证视频和文本特征维度")
    print("=" * 60)
    
    try:
        # 1. 导入必要的模块
        from models.base.few_shot import load, tokenize
        
        print("1. 加载CLIP RN50模型...")
        # 直接传入None作为cfg参数，因为在load函数中cfg主要用于某些特定配置
        model, preprocess = load("RN50", cfg=None, device="cpu")
        model.eval()
        print("   ✓ 模型加载成功")
        
        # 2. 测试文本编码
        print("\n2. 测试文本编码...")
        test_texts = [
            "a video of a person walking",
            "a video of a person running", 
            "a video of a person jumping"
        ]
        
        text_tokens = tokenize(test_texts).to("cpu")
        print(f"   文本tokens形状: {text_tokens.shape}")
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        
        print(f"   文本特征形状: {text_features.shape}")
        print(f"   文本特征维度: {text_features.shape[1]}")
        print("   ✓ 文本编码完成")
        
        # 3. 测试视频编码（模拟视频帧）
        print("\n3. 测试视频编码...")
        # 模拟视频帧：batch_size=2, channels=3, height=224, width=224
        batch_size = 2
        num_frames = 8
        channels = 3
        height = 224
        width = 224
        
        # 创建随机视频帧数据
        video_frames = torch.randn(batch_size, num_frames, channels, height, width)
        print(f"   模拟视频帧形状: {video_frames.shape}")
        
        # 将视频帧reshape为图像批次进行编码
        # [batch_size, num_frames, C, H, W] -> [batch_size*num_frames, C, H, W]
        frames_flat = video_frames.view(-1, channels, height, width)
        print(f"   展平后帧形状: {frames_flat.shape}")
        
        with torch.no_grad():
            # 编码所有帧
            frame_features = model.encode_image(frames_flat)
            
        print(f"   帧特征形状: {frame_features.shape}")
        print(f"   单帧特征维度: {frame_features.shape[1]}")
        
        # 重新组织为视频特征：[batch_size, num_frames, feature_dim]
        video_features = frame_features.view(batch_size, num_frames, -1)
        print(f"   视频特征形状: {video_features.shape}")
        print(f"   视频特征维度: {video_features.shape[2]}")
        print("   ✓ 视频编码完成")
        
        # 4. 验证维度匹配
        print("\n4. 验证维度匹配...")
        text_dim = text_features.shape[1]
        video_dim = video_features.shape[2]
        
        print(f"   文本特征维度: {text_dim}")
        print(f"   视频特征维度: {video_dim}")
        
        if text_dim == video_dim:
            print("   ✓ 维度匹配！文本和视频特征具有相同的维度")
            dimensions_match = True
        else:
            print("   ✗ 维度不匹配！")
            dimensions_match = False
        
        # 5. 测试相似度计算
        if dimensions_match:
            print("\n5. 测试相似度计算...")
            
            # 使用第一个视频的帧特征作为frame_features
            first_video = video_features[0]  # [num_frames, feature_dim]
            stage_text_features = text_features  # [num_texts, feature_dim]
            
            print(f"   frame_features形状: {first_video.shape}")
            print(f"   stage_text_features形状: {stage_text_features.shape}")
            
            # 使用cos_sim函数计算相似度（模拟第131行的代码）
            print("\n   使用cos_sim函数计算相似度矩阵...")
            similarity_matrix = cos_sim(first_video, stage_text_features)
            
            print(f"   similarity_matrix形状: {similarity_matrix.shape}")
            print(f"   预期形状: [num_frames={num_frames}, num_stages={len(test_texts)}]")
            print(f"   相似度范围: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
            print("   ✓ cos_sim相似度计算成功")
            
            # 对比使用F.normalize + torch.matmul的结果
            print("\n   对比使用F.normalize + torch.matmul的结果...")
            text_features_norm = F.normalize(text_features, p=2, dim=1)
            video_features_norm = F.normalize(first_video, p=2, dim=1)
            similarities_manual = torch.matmul(video_features_norm, text_features_norm.t())
            
            print(f"   手动计算相似度矩阵形状: {similarities_manual.shape}")
            print(f"   手动计算相似度范围: [{similarities_manual.min().item():.4f}, {similarities_manual.max().item():.4f}]")
            
            # 计算两种方法的差异
            diff = torch.abs(similarity_matrix - similarities_manual)
            print(f"   两种方法的最大差异: {diff.max().item():.6f}")
            
            # 显示具体的相似度值
            print("\n   详细cos_sim相似度矩阵:")
            print("   帧\\文本", end="")
            for i, text in enumerate(test_texts):
                print(f"\t文本{i+1}", end="")
            print()
            
            # 显示所有帧的相似度结果
            for frame_idx in range(num_frames):
                print(f"   帧{frame_idx+1:2d}", end="")
                for text_idx in range(len(test_texts)):
                    sim_val = similarity_matrix[frame_idx, text_idx].item()
                    print(f"\t{sim_val:.3f}", end="")
                print()
                
            # 验证第131行代码的维度匹配
            print(f"\n   ✓ 第131行代码验证:")
            print(f"     frame_features: {first_video.shape} -> [num_frames, feature_dim]")
            print(f"     stage_text_features: {stage_text_features.shape} -> [num_stages, feature_dim]")
            print(f"     similarity_matrix: {similarity_matrix.shape} -> [num_frames, num_stages]")
            print(f"     ✓ 维度匹配正确！")
        
        # 6. 测试模型内部参数
        print("\n6. 检查模型内部参数...")
        print(f"   text_projection形状: {model.text_projection.shape}")
        print(f"   transformer宽度: {model.transformer.width}")
        
        if hasattr(model.visual, 'output_dim'):
            print(f"   视觉编码器输出维度: {model.visual.output_dim}")
        
        # 7. 总结
        print("\n" + "=" * 60)
        print("验证结果总结:")
        print("=" * 60)
        print(f"✓ 文本特征维度: {text_dim}")
        print(f"✓ 视频特征维度: {video_dim}")
        if dimensions_match:
            print("✓ 维度匹配: 是")
            print("✓ 可以进行相似度计算: 是")
        else:
            print("✗ 维度匹配: 否")
            print("✗ 可以进行相似度计算: 否")
        print("=" * 60)
        
        return dimensions_match, text_dim, video_dim
        
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    verify_feature_dimensions()