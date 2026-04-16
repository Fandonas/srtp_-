import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import sys
import os

# 使用原始项目的模块
from models.base.few_shot import CNN_OTAM_CLIPFSAR, cos_sim, OTAM_cum_dist_v2
from models.base.few_shot import HEAD_REGISTRY
from einops import rearrange
from models.base.few_shot import extract_class_indices


@HEAD_REGISTRY.register()
class CNN_SEMANTIC_ALIGNMENT_FEW_SHOT(CNN_OTAM_CLIPFSAR):
    """
    小样本学习的语义对齐模型
    
    核心改进：
    1. 损失函数构成：第一个损失（视频-文本相似度） + 语义对齐损失（OTAM算法）
    2. 使用CLIP进行视频帧和文本的对齐
    3. 保持few-shot学习框架，使用support和target划分
    4. 帧级别语义对齐：每一帧都能与具体的语义阶段进行对齐，不进行帧分组
    
    语义对齐机制：
    - 将动作类别名称解析为多个语义阶段
    - 使用CLIP对每个语义阶段进行文本编码
    - 直接使用每一帧特征与语义阶段进行OTAM对齐
    - 找到帧与语义阶段的最优对齐路径
    """
    
    def __init__(self, cfg):
        super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).__init__(cfg)
        
        # 语义对齐损失权重
        self.semantic_loss_weight = getattr(cfg.TRAIN, 'SEMANTIC_LOSS_WEIGHT', 0.5)
        
        # 温度参数：用于控制语义对齐相似度的尺度（类似全监督模型）
        # 较小的temperature使相似度分布更加sharp，较大的使分布更加smooth
        self.semantic_temperature = getattr(cfg.TRAIN, 'SEMANTIC_TEMPERATURE', 0.1)
        
        # 延迟初始化：在第一次使用时才创建文本特征（避免初始化时CUDA不可用）
        self._full_clip_model = None
        self._semantic_stages = None
        self._stage_text_features = None
        self._text_features_initialized = False
        
    def _ensure_text_features_initialized(self):
        """确保文本特征已初始化（延迟初始化）"""
        if self._text_features_initialized:
            return
        
        try:
            # 获取完整的CLIP模型用于文本编码
            from models.base.few_shot import load
            
            # 确定设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._full_clip_model, _ = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device=device, cfg=self.args, jit=False)
            
            # 语义阶段解析
            self._semantic_stages = self._parse_semantic_stages()
            
            # 为每个阶段创建文本特征
            self._stage_text_features = self._create_stage_text_features()
            
            self._text_features_initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize text features: {e}")
            import traceback
            traceback.print_exc()
            # 设置空值，避免后续使用
            self._semantic_stages = {}
            self._stage_text_features = {}
            self._text_features_initialized = True  # 标记为已初始化，避免重复尝试
    
    def _parse_semantic_stages(self) -> Dict[str, List[str]]:
        """解析类别名称的语义阶段"""
        semantic_stages = {}
        
        # 合并训练和测试类别（few-shot中需要）
        all_classes = self.class_real_train + self.class_real_test
        
        for class_name in all_classes:
            # 使用逗号和"then"分割语义阶段
            stages = re.split(r'[,，]\s*|then\s+', class_name.lower())
            stages = [stage.strip() for stage in stages if stage.strip()]
            
            if len(stages) > 1:
                semantic_stages[class_name] = stages
            else:
                # 如果没有明确的分割，保持原始名称
                semantic_stages[class_name] = [class_name]
                
        return semantic_stages
    
    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:
        """为每个类别的语义阶段创建文本特征"""
        stage_text_features = {}
        
        # 导入tokenize函数
        from models.base.few_shot import tokenize
        
        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 使用torch.no_grad()避免梯度计算问题
        with torch.no_grad():
            for class_name, stages in self._semantic_stages.items():
                stage_features = []
                for stage in stages:
                    try:
                        # 使用CLIP编码每个阶段的文本
                        stage_text = f"a video of {stage}"#改
                        # 使用完整的CLIP模型进行文本编码
                        text_tokens = tokenize([stage_text]).to(device)
                        stage_feature = self._full_clip_model.encode_text(text_tokens)
                        stage_features.append(stage_feature)
                    except Exception as e:
                        print(f"Warning: Failed to encode stage '{stage}' for class '{class_name}': {e}")
                        continue
                
                if len(stage_features) > 0:
                    # 堆叠所有阶段特征 [num_stages, feature_dim]
                    stage_text_features[class_name] = torch.cat(stage_features, dim=0)
                else:
                    # 如果所有阶段编码失败，创建一个空特征
                    print(f"Warning: No valid features for class '{class_name}', skipping")
            
        return stage_text_features
    
    def _compute_semantic_alignment_loss(self, video_features: torch.Tensor, 
                                       class_names: List[str], debug=False) -> torch.Tensor:
        """
        计算语义对齐损失，使用OTAM算法进行帧级别的语义阶段对齐
        
        核心改进：
        1. 不对视频帧进行分组，直接使用每一帧
        2. 每一帧都能与具体的语义阶段进行对齐
        3. 使用OTAM算法找到帧与语义阶段的最优对齐路径
        
        Args:
            video_features: [batch_size, num_frames, feature_dim] 视频特征
            class_names: [batch_size] 类别名称列表
            
        Returns:
            semantic_loss: 语义对齐损失
        """
        try:
            # 确保文本特征已初始化
            self._ensure_text_features_initialized()
            
            if not self._stage_text_features or len(self._stage_text_features) == 0:
                # 如果文本特征未初始化，返回0损失
                return torch.tensor(0.0, device=video_features.device, requires_grad=True)
            
            batch_size, num_frames, feature_dim = video_features.shape
            
            # 使用列表收集所有损失，避免in-place操作
            semantic_losses = []
            
            for i in range(batch_size):
                class_name = class_names[i]
                
                # 获取该类别的语义阶段文本特征
                if class_name in self._stage_text_features:
                    stage_text_features = self._stage_text_features[class_name]  # [num_stages, feature_dim]
                    
                    # 确保stage_text_features在正确的设备上
                    stage_text_features = stage_text_features.to(video_features.device)
                    
                    # 直接使用每一帧，不进行分组
                    frame_features = video_features[i]  # [num_frames, feature_dim]
                    
                    # 计算每一帧与每个语义阶段的相似度矩阵
                    # 使用temperature参数控制相似度的尺度（类似全监督模型）
                    similarity_matrix = cos_sim(frame_features, stage_text_features) / self.semantic_temperature  # [num_frames, num_stages]
                    
                    # 调试信息：检查相似度矩阵的范围
                    if debug and i == 0:
                        print(f"Debug - Similarity matrix range (after temperature): [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
                        print(f"Debug - Semantic temperature: {self.semantic_temperature}")
                    
                    # 使用OTAM算法找到帧与语义阶段的最优对齐路径
                    dists = 1 - torch.sigmoid(similarity_matrix)  # 使用sigmoid归一化后再转换为距离（越小越好）
                    
                    # 将2D距离矩阵转换为4D张量：[1, 1, num_frames, num_stages]
                    dists_4d = dists.unsqueeze(0).unsqueeze(0)  # [1, 1, num_frames, num_stages]
                    cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [1, 1]
                    
                    # 调试信息：检查OTAM输出
                    if debug and i == 0:
                        print(f"Debug - OTAM output: {cum_dists[0, 0].item():.4f}")
                    
                    # 语义对齐损失：最小化帧与语义阶段的最优对齐路径总距离
                    semantic_loss = torch.abs(cum_dists[0, 0])  # 取OTAM距离的绝对值
                    semantic_losses.append(semantic_loss)
            
            if len(semantic_losses) > 0:
                # 使用torch.stack避免in-place操作
                all_losses = torch.stack(semantic_losses)
                return all_losses.mean()
            else:
                return torch.tensor(0.0, device=video_features.device, requires_grad=True)
                
        except Exception as e:
            print(f"Warning: Semantic alignment loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)
    
    def forward(self, inputs):
        """
        Few-shot学习的前向传播
        
        Args:
            inputs: 包含以下键的字典
                - support_set: [support_size, num_frames, C, H, W]
                - support_labels: [support_size] few-shot任务内的标签
                - target_set: [target_size, num_frames, C, H, W]
                - real_support_labels: [support_size] 真实类别标签（索引）
        """
        support_images, support_labels, target_images, support_real_class = \
            inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']
        
        # 调用父类的forward获取基础输出
        model_dict = super().forward(inputs)
        
        # 获取视频特征用于语义对齐（仅在训练模式下计算，避免不必要计算）
        try:
            if self.training:
                # ===== 关键改进：使用与父类forward相同的特征增强流程 =====
                # 1. 获取原始Visual Encoder特征
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                # 2. Query特征：通过Temporal Transformer增强
                target_features_enhanced = self.context2(target_features, target_features, target_features)
                
                # 3. Support特征：拼接Text + Temporal Transformer增强（与父类forward逻辑一致）
                # 获取text encoder特征
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # [support_bs, 1, dim]
                context_support = self.mid_layer(context_support)
                
                # 如果配置了MERGE_BEFORE，先聚合同类别的support
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features_merged = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features_merged = torch.stack(support_features_merged)
                    context_support_merged = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support_merged = torch.stack(context_support_merged)
                    
                    # 拼接text到visual特征
                    support_features_with_text = torch.cat([support_features_merged, context_support_merged], dim=1)  # [num_classes, frames+1, dim]
                    # 通过Temporal Transformer融合
                    support_features_enhanced = self.context2(support_features_with_text, support_features_with_text, support_features_with_text)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                    
                    # 为了语义对齐损失，需要将增强后的特征扩展回原始batch大小
                    # 每个类别对应的样本使用相同的增强后的类别特征
                    support_features_for_semantic = []
                    for i in range(support_bs):
                        label = support_labels[i]
                        class_idx = (unique_labels == label).nonzero(as_tuple=True)[0][0]
                        support_features_for_semantic.append(support_features_enhanced[class_idx])
                    support_features_enhanced = torch.stack(support_features_for_semantic)
                else:
                    # 直接拼接text到每个support样本
                    support_features_with_text = torch.cat([support_features, context_support], dim=1)  # [support_bs, frames+1, dim]
                    # 通过Temporal Transformer融合
                    support_features_enhanced = self.context2(support_features_with_text, support_features_with_text, support_features_with_text)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                
                # 4. 合并增强后的support和target特征用于语义对齐损失计算
                all_features = torch.cat([support_features_enhanced, target_features_enhanced], dim=0)  # [support_bs+target_bs, num_frames, dim]
                
                # 获取对应的类别名称（用于语义对齐）
                all_class_names = []
                
                # Support部分的类别名称
                for i in range(support_bs):
                    label_idx = int(support_real_class[i].item())  # 转换为整数
                    if label_idx < len(self.class_real_train):
                        all_class_names.append(self.class_real_train[label_idx])
                    else:
                        test_idx = label_idx - len(self.class_real_train)
                        if test_idx >= 0 and test_idx < len(self.class_real_test):
                            all_class_names.append(self.class_real_test[test_idx])
                        else:
                            # 安全默认值
                            all_class_names.append(self.class_real_train[0] if len(self.class_real_train) > 0 else "")
                
                # Target部分的类别名称（few-shot中target来自support类别集合）
                # 需要从task_dict中获取real_target_labels来确定target的类别
                # 这里先使用support的类别，如果inputs中有real_target_labels则使用它
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        label_idx = int(target_real_labels[i].item())  # 转换为整数
                        if label_idx < len(self.class_real_train):
                            all_class_names.append(self.class_real_train[label_idx])
                        else:
                            test_idx = label_idx - len(self.class_real_train)
                            if test_idx >= 0 and test_idx < len(self.class_real_test):
                                all_class_names.append(self.class_real_test[test_idx])
                            else:
                                all_class_names.append(self.class_real_train[0] if len(self.class_real_train) > 0 else "")
                else:
                    # 如果没有real_target_labels，假设target使用support的类别
                    for i in range(min(target_bs, support_bs)):
                        label_idx = int(support_real_class[i].item())  # 转换为整数
                        if label_idx < len(self.class_real_train):
                            all_class_names.append(self.class_real_train[label_idx])
                        else:
                            test_idx = label_idx - len(self.class_real_train)
                            if test_idx >= 0 and test_idx < len(self.class_real_test):
                                all_class_names.append(self.class_real_test[test_idx])
                            else:
                                all_class_names.append(self.class_real_train[0] if len(self.class_real_train) > 0 else "")
                
                # 如果target数量超过已处理的类别名称
                if target_bs > len(all_class_names) - support_bs:
                    last_class = all_class_names[-1] if len(all_class_names) > 0 else (self.class_real_train[0] if len(self.class_real_train) > 0 else "")
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for i in range(needed):
                        all_class_names.append(last_class)
                
                # 计算语义对齐损失
                debug_semantic = hasattr(self, '_debug_count') and self._debug_count < 3
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                if debug_semantic:
                    self._debug_count += 1
                    
                semantic_alignment_loss = self._compute_semantic_alignment_loss(
                    all_features, all_class_names, debug=debug_semantic
                )
                
                # 将语义对齐损失添加到输出字典
                model_dict['semantic_alignment_loss'] = semantic_alignment_loss
            else:
                # 评估模式下不计算语义对齐损失（可选）
                model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)
        except Exception as e:
            # 如果计算语义对齐损失时出错，记录警告但不中断训练
            print(f"Warning: Failed to compute semantic alignment loss: {e}")
            import traceback
            traceback.print_exc()
            model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)
        
        return model_dict
    
    def loss(self, task_dict, model_dict):
        """
        计算总损失：原始few-shot损失 + 语义对齐损失
        
        损失构成说明：
        1. base_loss: Few-shot分类损失 (support-query对齐)
        2. classification_loss: 视觉-文本分类损失 (如果启用)
        3. semantic_alignment_loss: 语义对齐损失 (帧-语义阶段对齐)
        
        Args:
            task_dict: 包含标签的字典
            model_dict: 包含模型输出的字典
            
        Returns:
            dict: 包含各个损失项的字典（与全监督模型格式一致）
        """
        # 基础few-shot损失（父类的损失）
        base_loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        
        # 如果启用了分类损失，也加上
        classification_loss = torch.tensor(0.0, device=base_loss.device)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if model_dict.get("class_logits") is not None:
                all_labels = torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0)
                classification_loss = F.cross_entropy(model_dict["class_logits"], all_labels.long())
                base_loss = base_loss + self.args.TRAIN.USE_CLASSIFICATION_VALUE * classification_loss
        
        # 语义对齐损失
        semantic_loss = model_dict.get("semantic_alignment_loss", torch.tensor(0.0, device=base_loss.device))
        
        # 总损失
        total_loss = base_loss + self.semantic_loss_weight * semantic_loss
        
        # 应用batch size缩放（如果配置了）
        if hasattr(self.args.TRAIN, 'BATCH_SIZE'):
            total_loss = total_loss / self.args.TRAIN.BATCH_SIZE
        
        # 返回所有损失组件（与全监督模型格式完全一致）
        return {
            'total_loss': total_loss,
            'text_similarity_loss': base_loss,  # few-shot中对应base_loss
            'semantic_loss': semantic_loss,
            'weighted_semantic_loss': self.semantic_loss_weight * semantic_loss
        }

