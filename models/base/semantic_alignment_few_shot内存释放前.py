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

    核心功能：
    1. 帧对齐：使用OTAM算法进行support和query视频的帧级别对齐
    2. 语义对齐：将视频帧与动作语义阶段进行对齐
    3. 损失函数构成：帧对齐损失（few-shot分类损失） + 语义对齐损失（OTAM算法）
    4. 保持few-shot学习框架，使用support和target划分

    语义对齐机制：
    - 将动作类别名称解析为多个语义阶段
    - 使用CLIP对每个语义阶段进行文本编码
    - 直接使用每一帧特征与语义阶段进行OTAM对齐
    - 找到帧与语义阶段的最优对齐路径

    特点：不使用类别级别的文本对齐，只保留帧对齐和语义对齐
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
        
    def _get_class_name(self, label_idx: int) -> str:
        """
        根据label_idx和当前训练状态获取正确的类别名称

        重要说明：
        - 训练集和测试集的ID空间是独立的，都从0开始
        - 训练时(self.training=True): label_idx ∈ [0, 30] → class_real_train[label_idx]
        - 测试时(self.training=False): label_idx ∈ [0, 9] → class_real_test[label_idx]

        Args:
            label_idx: 类别索引（来自real_support_labels或real_target_labels）

        Returns:
            对应的类别名称字符串
        """
        label_idx = int(label_idx)

        # 根据模型的training状态选择对应的类别列表
        if self.training:
            # 训练模式：使用训练集类别列表
            if 0 <= label_idx < len(self.class_real_train):
                return self.class_real_train[label_idx]
            else:
                # 索引超出范围，返回默认值
                print(f"Warning: label_idx {label_idx} out of range for train classes (0-{len(self.class_real_train)-1})")
                return self.class_real_train[0] if len(self.class_real_train) > 0 else ""
        else:
            # 评估/测试模式：使用测试集类别列表
            if 0 <= label_idx < len(self.class_real_test):
                return self.class_real_test[label_idx]
            else:
                # 索引超出范围，返回默认值
                print(f"Warning: label_idx {label_idx} out of range for test classes (0-{len(self.class_real_test)-1})")
                return self.class_real_test[0] if len(self.class_real_test) > 0 else ""

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
            self._stage_text_features = self._create_stage_text_features()#提取了整个数据集的文本特征

            self._text_features_initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize text features: {e}")
            import traceback
            traceback.print_exc()
            # 设置空值，避免后续使用
            self._semantic_stages = {}
            self._stage_text_features = {}
            self._text_features_initialized = True  # 标记为已初始化，避免重复尝试
    
    def _parse_semantic_stages(self) -> Dict[str, List[str]]:#分割阶段"," ", " ", then "
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
    
    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:#分阶段提取特征后cat一起
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
                        stage_text = f"a video of {stage}"
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

    def _compute_semantic_alignment_loss_multi_class(self,
                                                     video_features: torch.Tensor,
                                                     class_names: List[str],
                                                     debug: bool = False) -> torch.Tensor:
        """
        多类别语义对齐版本：
        - 对于每个视频，与当前 episode 中出现的每一个标签的语义阶段序列做 OTAM 对齐；
        - 得到按类别的累积距离向量 cum_dists[class]；
        - 使用 -cum_dists 作为 logits，经 softmax / CrossEntropy 与真实标签构成语义分类损失；
        - 保留 temperature 缩放和 sigmoid 归一化：cos_sim / T → sigmoid → 1 - sigmoid。

        Args:
            video_features: [batch_size, num_frames, feature_dim]，support 和 target 特征拼接
            class_names:    [batch_size]，与 video_features 一一对应的真实类别名称
        """
        try:
            # 确保文本特征已初始化
            self._ensure_text_features_initialized()

            device = video_features.device

            if not self._stage_text_features or len(self._stage_text_features) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            batch_size, num_frames, feature_dim = video_features.shape

            # 该 episode 中实际出现的类别（按出现顺序去重），只在这些标签之间做多类别对齐
            episode_classes: List[str] = []
            for name in class_names:
                if (name in self._stage_text_features) and (name not in episode_classes):
                    episode_classes.append(name)

            if len(episode_classes) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # 预先把本  用到的阶段文本特征搬到正确设备上
            episode_stage_feats = {
                cname: self._stage_text_features[cname].to(device)
                for cname in episode_classes
            }#当前episode的每个类的特征

            semantic_losses: List[torch.Tensor] = []

            for i in range(batch_size):
                gt_name = class_names[i]
                if gt_name not in episode_stage_feats:
                    # 没有该类别的语义阶段特征时，跳过该样本
                    continue

                frame_features = video_features[i]  # [num_frames, feature_dim]
                class_cum_dists = []

                # 让该视频与 episode 中的每一个标签做 OTAM 对齐
                for cls_idx, cname in enumerate(episode_classes):
                    stage_text_features = episode_stage_feats[cname]  # [num_stages, feature_dim]

                    # 计算帧与语义阶段的相似度矩阵，并做 temperature 缩放
                    similarity_matrix = cos_sim(frame_features, stage_text_features) / self.semantic_temperature

                    if debug and i == 0 and cls_idx == 0:
                        print(
                            f"Debug(multi) - Similarity matrix range (after temperature): "
                            f"[{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]"
                        )
                        print(f"Debug(multi) - Semantic temperature: {self.semantic_temperature}")

                    # 使用 sigmoid 归一化后转成距离：相似度越高 -> 距离越小
                    dists = 1 - torch.sigmoid(similarity_matrix)  # [num_frames, num_stages]

                    # OTAM 要求 4D：[tb, sb, ts, ss]，这里单个视频 / 单个标签 => tb=1, sb=1
                    dists_4d = dists.unsqueeze(0).unsqueeze(0)  # [1, 1, num_frames, num_stages]
                    cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [1, 1]

                    if debug and i == 0 and cls_idx == 0:
                        print(f"Debug(multi) - OTAM output (single class): {cum_dists[0, 0].item():.4f}")

                    class_cum_dists.append(cum_dists[0, 0])

                if len(class_cum_dists) == 0:
                    continue

                class_cum_dists = torch.stack(class_cum_dists)  # [num_episode_classes]

                # 以 -cum_dists 作为 logits，距离越小 -> 概率越大
                logits = -class_cum_dists  # [C]

                # 找到真实标签在 episode_classes 中的下标
                gt_index = episode_classes.index(gt_name)
                target_idx = torch.tensor(gt_index, device=device, dtype=torch.long)#生成的是 0 维标量张量，值为gt_index

                # CrossEntropy(logits, gt) 等价于 softmax(-cum_dists) 后取 -log p(gt)
                loss_i = F.cross_entropy(logits.unsqueeze(0), target_idx.unsqueeze(0))
                semantic_losses.append(loss_i)

            if len(semantic_losses) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            return torch.stack(semantic_losses).mean()

        except Exception as e:
            print(f"Warning: Multi-class semantic alignment loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

    def _fuse_semantic_and_visual_probs_eval(self, inputs, model_dict):
        """
        仅在评估模式下调用：
        使用“语义对齐概率 + 帧对齐概率”作为最终分类结果，
        融合方式模仿文本对齐 + 帧对齐的 COMBINE 分支：

            p_fused ∝ p_semantic^β * p_visual^(1-β)
            logits = -p_fused

        其中：
        - p_visual 来自当前 model_dict["logits"] 的 softmax（few-shot OTAM 帧对齐）
        - p_semantic 来自“目标视频 vs episode 内每个标签的语义阶段序列”的 OTAM 对齐，
          概率计算方式参照文本对齐：对 -cum_dists 做 softmax。

        参与概率计算的视频与文本对齐一致：只使用当前 episode 的 query/target 视频。
        """
        # 只在测试模式并且显式打开 SEMANTIC_COMBINE 时生效
        if self.training:
            return model_dict
        if not hasattr(self.args.TRAIN, "SEMANTIC_COMBINE") or not self.args.TRAIN.SEMANTIC_COMBINE:
            return model_dict
        if "logits" not in model_dict:
            return model_dict

        try:
            """
                    Few-shot学习的前向传播

                    Args:
                        inputs: 包含以下键的字典
                            - support_set: [support_size, num_frames, C, H, W]
                            - support_labels: [support_size] few-shot任务内的标签
                            - target_set: [target_size, num_frames, C, H, W]
                            - real_support_labels: [support_size] 真实类别标签（索引）
            """
            support_images = inputs["support_set"]
            support_labels = inputs["support_labels"]
            target_images = inputs["target_set"]
            support_real_class = inputs["real_support_labels"]

            # 先获取视觉 backbone 的特征（与文本对齐、语义对齐保持一致）
            # 这里 get_feats 在 eval 模式下会使用 text_features_test 构建 support_features_text，
            # 但我们这里只需要 support_features 和 target_features。
            support_features, target_features, _ = self.get_feats(
                support_images, target_images, support_real_class
            )

            device = target_features.device
            target_bs = target_features.shape[0]
            if target_bs == 0:
                return model_dict

            # 从 support_labels / support_real_class 构建 episode 内的类别名字（顺序与 OTAM 类别顺序一致）
            unique_labels = torch.unique(support_labels)
            episode_class_names: List[str] = []
            for c in unique_labels:
                # 取该 episodic label 对应的第一个 support 样本，找到其真实类别 id
                idx = extract_class_indices(support_labels, c)[0]
                label_idx = int(support_real_class[idx].item())
                # 使用新的辅助方法，根据training状态获取正确的类别名称
                cname = self._get_class_name(label_idx)
                episode_class_names.append(cname)

            num_classes = len(episode_class_names)
            if num_classes == 0:
                return model_dict

            # 视觉分支概率：直接对当前 few-shot OTAM logits 做 softmax
            visual_logits = model_dict["logits"]
            if visual_logits.shape[0] != target_bs or visual_logits.shape[1] != num_classes:
                # 形状不匹配时，不做融合以避免错误
                return model_dict
            visual_probs = F.softmax(visual_logits, dim=1)  # [target_bs, num_classes]

            # 语义分支：构建阶段文本特征，并对每个 target 视频与每个类别做 OTAM 对齐
            self._ensure_text_features_initialized()
            if not self._stage_text_features or len(self._stage_text_features) == 0:
                return model_dict

            episode_stage_feats = {}
            for cname in episode_class_names:
                if cname in self._stage_text_features:
                    episode_stage_feats[cname] = self._stage_text_features[cname].to(device)
            if len(episode_stage_feats) != num_classes:
                # 某些类别没有语义阶段特征，跳过融合
                return model_dict

            semantic_cum = target_features.new_zeros(target_bs, num_classes)
            for qi in range(target_bs):
                frame_features = target_features[qi]  # [num_frames, dim]
                for cj, cname in enumerate(episode_class_names):
                    stage_text_features = episode_stage_feats[cname]  # [num_stages, dim]
                    similarity_matrix = cos_sim(frame_features, stage_text_features) / self.semantic_temperature
                    dists = 1 - torch.sigmoid(similarity_matrix)
                    d4 = dists.unsqueeze(0).unsqueeze(0)
                    cum = OTAM_cum_dist_v2(d4, lbda=0.5)[0, 0]  # 标量
                    semantic_cum[qi, cj] = cum

            # 语义对齐概率：参照文本对齐的做法（few_shot.py:2894）
            # 文本对齐：softmax(similarity)，语义对齐：softmax(-distance)，将距离转为相似度后 softmax
            semantic_probs = F.softmax(-semantic_cum, dim=1)  # [target_bs, num_classes]

            # 融合：借鉴 few_shot.py:2952-2954 + 3016 的两次取负号设计
            # 语义对齐（参照文本对齐角色）占 β 权重，视觉对齐（帧对齐角色）占 1-β 权重
            if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
                # 第一次取负号
                fused_dists = -(semantic_probs.pow(self.args.TRAIN.TEXT_COFF) * visual_probs.pow(1.0 - self.args.TRAIN.TEXT_COFF))
            else:
                fused_dists = -(semantic_probs.pow(0.5) * visual_probs.pow(0.5))

            # 第二次取负号：借鉴 few_shot.py:3016 的 return {'logits': -class_dists}
            # 两次负号抵消，最终 logits = semantic_probs^β * visual_probs^(1-β)
            model_dict["logits"] = -fused_dists
            return model_dict

        except Exception as e:
            print(f"Warning: semantic-visual fusion in eval failed: {e}")
            import traceback
            traceback.print_exc()
            return model_dict
    
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
                # ===== 只使用视觉特征进行增强，不使用文本对齐 =====
                # 1. 获取原始Visual Encoder特征
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)# [support_bs, NUM_INPUT_FRAMES, dim]
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]

                # 2. Query特征：通过Temporal Transformer增强
                target_features_enhanced = self.context2(target_features, target_features, target_features)
                #改加入文本qery
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # [support_bs, 1, dim]
                context_support = self.mid_layer(context_support)
                # 3. Support特征：通过Temporal Transformer增强（不使用文本特征）
                # 如果配置了MERGE_BEFORE，先聚合同类别的support
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features_merged = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features_merged = torch.stack(support_features_merged)

                    # 通过Temporal Transformer融合（不使用文本特征）
                    support_features_enhanced = self.context2(support_features_merged, support_features_merged, support_features_merged)

                    # 为了语义对齐损失，需要将增强后的特征扩展回原始batch大小
                    # 每个类别对应的样本使用相同的增强后的类别特征
                    support_features_for_semantic = []
                    for i in range(support_bs):
                        label = support_labels[i]
                        class_idx = (unique_labels == label).nonzero(as_tuple=True)[0][0]
                        support_features_for_semantic.append(support_features_enhanced[class_idx])
                    support_features_enhanced = torch.stack(support_features_for_semantic)
                else:
                    # 直接对每个support样本进行Temporal Transformer增强（不使用文本特征）
                    support_features_enhanced = self.context2(support_features, support_features, support_features)
                
                # 4. 合并增强后的support和target特征用于语义对齐损失计算
                all_features = torch.cat([support_features_enhanced, target_features_enhanced], dim=0)  # [support_bs+target_bs, num_frames, dim]
                
                # 获取对应的类别名称（用于语义对齐）
                all_class_names = []
                
                # Support部分的类别名称
                for i in range(support_bs):
                    label_idx = int(support_real_class[i].item())
                    # 使用辅助方法获取正确的类别名称
                    all_class_names.append(self._get_class_name(label_idx))

                # Target部分的类别名称（few-shot中target来自support类别集合）
                # 需要从task_dict中获取real_target_labels来确定target的类别
                # 这里先使用support的类别，如果inputs中有real_target_labels则使用它
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        label_idx = int(target_real_labels[i].item())
                        # 使用辅助方法获取正确的类别名称
                        all_class_names.append(self._get_class_name(label_idx))
                else:
                    # 如果没有real_target_labels，假设target使用support的类别
                    for i in range(min(target_bs, support_bs)):
                        label_idx = int(support_real_class[i].item())
                        # 使用辅助方法获取正确的类别名称
                        all_class_names.append(self._get_class_name(label_idx))
                
                # 如果target数量超过已处理的类别
                if target_bs > len(all_class_names) - support_bs:
                    # 使用辅助方法获取默认类别名称（根据training状态）
                    if len(all_class_names) > 0:
                        last_class = all_class_names[-1]
                    else:
                        # 根据training状态选择默认值
                        if self.training and len(self.class_real_train) > 0:
                            last_class = self.class_real_train[0]
                        elif not self.training and len(self.class_real_test) > 0:
                            last_class = self.class_real_test[0]
                        else:
                            last_class = ""
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for i in range(needed):
                        all_class_names.append(last_class)
                
                # 计算语义对齐损失
                debug_semantic = hasattr(self, '_debug_count') and self._debug_count < 3
                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                if debug_semantic:
                    self._debug_count += 1
                    
                semantic_alignment_loss = self._compute_semantic_alignment_loss_multi_class(
                    all_features, all_class_names, debug=debug_semantic
                )
                
                # 将语义对齐损失添加到输出字典
                model_dict['semantic_alignment_loss'] = semantic_alignment_loss
            else:
                # ===== 评估模式：只使用视觉特征进行增强，不使用文本对齐 =====
                support_features, target_features, _ = self.get_feats(support_images, target_images, support_labels)
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]

                # Query特征：通过Temporal Transformer增强
                target_features_enhanced = self.context2(target_features, target_features, target_features)

                # Support特征：通过Temporal Transformer增强（不使用文本特征）
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features_merged = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features_merged = torch.stack(support_features_merged)

                    # 通过Temporal Transformer融合（不使用文本特征）
                    support_features_enhanced = self.context2(support_features_merged, support_features_merged, support_features_merged)

                    # 扩展回原始batch大小
                    support_features_for_semantic = []
                    for i in range(support_bs):
                        label = support_labels[i]
                        class_idx = (unique_labels == label).nonzero(as_tuple=True)[0][0]
                        support_features_for_semantic.append(support_features_enhanced[class_idx])
                    support_features_enhanced = torch.stack(support_features_for_semantic)
                else:
                    # 直接对每个support样本进行Temporal Transformer增强（不使用文本特征）
                    support_features_enhanced = self.context2(support_features, support_features, support_features)

                    # 聚合同类别的support
                    unique_labels = torch.unique(support_labels)
                    support_features_enhanced = [torch.mean(torch.index_select(support_features_enhanced, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features_enhanced = torch.stack(support_features_enhanced)

                    # 扩展回原始batch大小用于语义对齐
                    support_features_for_semantic = []
                    for i in range(support_bs):
                        label = support_labels[i]
                        class_idx = (unique_labels == label).nonzero(as_tuple=True)[0][0]
                        support_features_for_semantic.append(support_features_enhanced[class_idx])
                    support_features_enhanced = torch.stack(support_features_for_semantic)
                
                # 合并增强后的support和target特征用于语义对齐损失计算
                all_features = torch.cat([support_features_enhanced, target_features_enhanced], dim=0)
                
                # 获取对应的类别名称（用于语义对齐）
                all_class_names = []
                
                # Support部分的类别名称
                for i in range(support_bs):
                    label_idx = int(support_real_class[i].item())
                    # 使用辅助方法获取正确的类别名称
                    all_class_names.append(self._get_class_name(label_idx))

                # Target部分的类别名称
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        label_idx = int(target_real_labels[i].item())
                        # 使用辅助方法获取正确的类别名称
                        all_class_names.append(self._get_class_name(label_idx))
                else:
                    for i in range(min(target_bs, support_bs)):
                        label_idx = int(support_real_class[i].item())
                        # 使用辅助方法获取正确的类别名称
                        all_class_names.append(self._get_class_name(label_idx))
                
                # 如果target数量超过已处理的类别名称
                if target_bs > len(all_class_names) - support_bs:
                    # 使用辅助方法获取默认类别名称（根据training状态）
                    if len(all_class_names) > 0:
                        last_class = all_class_names[-1]
                    else:
                        # 根据training状态选择默认值
                        if self.training and len(self.class_real_train) > 0:
                            last_class = self.class_real_train[0]
                        elif not self.training and len(self.class_real_test) > 0:
                            last_class = self.class_real_test[0]
                        else:
                            last_class = ""
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for i in range(needed):
                        all_class_names.append(last_class)
                
                # 计算语义对齐损失（用于评估分析，不用于反向传播）
                with torch.no_grad():
                    semantic_alignment_loss = self._compute_semantic_alignment_loss_multi_class(
                        all_features, all_class_names, debug=False
                    )
                
                # 将语义对齐损失添加到输出字典（用于评估和日志记录）
                model_dict['semantic_alignment_loss'] = semantic_alignment_loss
        except Exception as e:
            # 如果计算语义对齐损失时出错，记录警告但不中断训练
            print(f"Warning: Failed to compute semantic alignment loss: {e}")
            import traceback
            traceback.print_exc()
            model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)

        # 评估模式下，可选地用“语义对齐概率 + 帧对齐概率”的融合结果替换 logits
        model_dict = self._fuse_semantic_and_visual_probs_eval(inputs, model_dict)

        return model_dict
    
    # def loss(self, task_dict, model_dict):
    #     """
    #     计算总损失：原始few-shot损失 + 语义对齐损失
    #
    #     损失构成说明：
    #     1. base_loss: Few-shot分类损失 (帧对齐，support-query对齐)
    #     2. semantic_alignment_loss: 语义对齐损失 (帧-语义阶段对齐)
    #
    #     Args:
    #         task_dict: 包含标签的字典
    #         model_dict: 包含模型输出的字典
    #
    #     Returns:
    #         dict: 包含各个损失项的字典
    #     """
    #     # 基础few-shot损失（父类的帧对齐损失）
    #     base_loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    #
    #     # 语义对齐损失
    #     semantic_loss = model_dict.get("semantic_alignment_loss", torch.tensor(0.0, device=base_loss.device))
    #
    #     # 总损失
    #     total_loss = base_loss + self.semantic_loss_weight * semantic_loss
    #
    #     # 应用batch size缩放（如果配置了）
    #     if hasattr(self.args.TRAIN, 'BATCH_SIZE'):
    #         total_loss = total_loss / self.args.TRAIN.BATCH_SIZE
    #
    #     # 返回所有损失组件
    #     return {
    #         'total_loss': total_loss,
    #         'frame_alignment_loss': base_loss,  # few-shot帧对齐损失
    #         'semantic_loss': semantic_loss,
    #         'weighted_semantic_loss': self.semantic_loss_weight * semantic_loss
    #     }
