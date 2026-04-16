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
        # True→原始行为(行=帧)；False→转置行为(行=阶段，一阶段多帧)
        self.semantic_transpose = getattr(cfg.TRAIN, 'SEMANTIC_TRANSPOSE', False)
        
        # 延迟初始化：在第一次使用时才创建文本特征（避免初始化时CUDA不可用）
        self._full_clip_model = None
        self._semantic_stages = None
        self._stage_text_features = None
        self._text_features_initialized = False

        # 可学习的训练类 query_embed：用冻结的 CLIP 文本特征初始化，训练时随梯度更新
        # 形状 [num_train_classes, mid_dim]，参数量极小（如64×512=32K）
        # 测试阶段仍使用冻结的 text_features_test（novel类无法预训练）
        num_train_classes = self.text_features_train.shape[0]
        self.query_embed_train = nn.Embedding(num_train_classes, self.mid_dim)
        self.query_embed_train.weight = nn.Parameter(
            self.text_features_train.detach().clone()
        )

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

            # 文本特征已全部缓存到 _stage_text_features，_full_clip_model 不再需要，立即释放显存
            del self._full_clip_model
            self._full_clip_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

                    # SEMANTIC_TRANSPOSE=False: 转置为[1,1,num_stages,num_frames]，行=阶段，一阶段对多帧
                    # SEMANTIC_TRANSPOSE=True:  原始[1,1,num_frames,num_stages]，行=帧，一帧对多阶段
                    if not self.semantic_transpose:
                        dists_4d = dists.T.unsqueeze(0).unsqueeze(0)  # [1, 1, num_stages, num_frames]
                    else:
                        dists_4d = dists.unsqueeze(0).unsqueeze(0)    # [1, 1, num_frames, num_stages]
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
        多类别语义对齐版本（向量化实现）：
        - 对于每个视频，与当前 episode 中出现的每一个标签的语义阶段序列做 OTAM 对齐；
        - 得到按类别的累积距离向量 cum_dists[class]；
        - 使用 -cum_dists 作为 logits，经 softmax / CrossEntropy 与真实标签构成语义分类损失；
        - 保留 temperature 缩放和 sigmoid 归一化：cos_sim / T → sigmoid → 1 - sigmoid。

        向量化优化：将原来的 batch_size × num_classes 双重循环改为按类别逐次批量处理，
        OTAM 调用次数从 batch_size×num_classes 次降至 num_classes 次，
        数学结果与原始逐个循环完全等价。

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

            # 预先把本 episode 用到的阶段文本特征搬到正确设备上
            episode_stage_feats = {
                cname: self._stage_text_features[cname].to(device)
                for cname in episode_classes
            }  # 当前 episode 每个类的文本特征

            # 有效样本掩码：gt_name 存在于 episode_stage_feats 的视频才参与损失计算
            valid_indices = [i for i in range(batch_size) if class_names[i] in episode_stage_feats]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # 将视频特征展平为 [batch_size * num_frames, feature_dim]，
            # 供后续与各类别阶段特征批量计算余弦相似度
            frames_flat = video_features.reshape(batch_size * num_frames, feature_dim)

            # ===== 向量化核心：按类别逐次批量处理（num_classes 次，替代原来的 batch×classes 次）=====
            # 对每个类别 c，一次性计算所有 batch_size 个视频与该类别的 OTAM 累积距离
            all_cum_dists_list: List[torch.Tensor] = []

            for cls_idx, cname in enumerate(episode_classes):
                stage_text_features = episode_stage_feats[cname]  # [num_stages, feature_dim]
                num_stages = stage_text_features.shape[0]

                # 批量计算所有视频帧与当前类别语义阶段的余弦相似度
                # cos_sim: [batch_size*num_frames, D] × [num_stages, D] → [batch_size*num_frames, num_stages]
                sim_flat = cos_sim(frames_flat, stage_text_features) / self.semantic_temperature
                # 还原为 [batch_size, num_frames, num_stages]
                # 等价于原循环中对每个 i 单独计算的 [num_frames, num_stages]，数值完全一致
                similarity_matrix = sim_flat.reshape(batch_size, num_frames, num_stages)

                if debug and cls_idx == 0:
                    # 打印第一个视频的相似度范围，与原 i==0, cls_idx==0 等价
                    sm0 = similarity_matrix[0]
                    print(
                        f"Debug(multi) - Similarity matrix range (after temperature): "
                        f"[{sm0.min().item():.4f}, {sm0.max().item():.4f}]"
                    )
                    print(f"Debug(multi) - Semantic temperature: {self.semantic_temperature}")

                # sigmoid 归一化后转距离：相似度越高 -> 距离越小
                dists = 1 - torch.sigmoid(similarity_matrix)  # [batch_size, num_frames, num_stages]

                # SEMANTIC_TRANSPOSE=False: 转置为[B,1,num_stages,num_frames]，行=阶段，一阶段对多帧
                # SEMANTIC_TRANSPOSE=True:  原始[B,1,num_frames,num_stages]，行=帧，一帧对多阶段
                if not self.semantic_transpose:
                    dists_4d = dists.transpose(-1, -2).unsqueeze(1)  # [batch_size, 1, num_stages, num_frames]
                else:
                    dists_4d = dists.unsqueeze(1)                     # [batch_size, 1, num_frames, num_stages]
                cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [batch_size, 1]

                if debug and cls_idx == 0:
                    print(f"Debug(multi) - OTAM output (first video, first class): {cum_dists[0, 0].item():.4f}")

                all_cum_dists_list.append(cum_dists.squeeze(1))  # [batch_size]

            # 拼成 [batch_size, num_episode_classes]
            # all_cum_dists[i, c] 等价于原循环中视频 i 对类别 c 的 OTAM 累积距离
            all_cum_dists = torch.stack(all_cum_dists_list, dim=1)  # [batch_size, C]
            logits_all = -all_cum_dists  # [batch_size, C]，距离越小 -> logit 越大

            # 构建有效样本的真实类别索引
            gt_indices = torch.tensor(
                [episode_classes.index(class_names[i]) for i in valid_indices],
                device=device, dtype=torch.long
            )

            # 批量 CrossEntropy（reduction='mean'）等价于原来逐样本计算后取均值
            loss = F.cross_entropy(logits_all[valid_indices], gt_indices)
            return loss

        except Exception as e:
            print(f"Warning: Multi-class semantic alignment loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

    def _fuse_semantic_and_visual_probs_eval(self, inputs, model_dict, cached_target_features=None):
        """
        仅在评估模式下调用：
        使用"语义对齐概率 + 帧对齐概率"作为最终分类结果，
        融合方式模仿文本对齐 + 帧对齐的 COMBINE 分支：

            p_fused ∝ p_semantic^β * p_visual^(1-β)
            logits = -p_fused

        其中：
        - p_visual 来自当前 model_dict["logits"] 的 softmax（few-shot OTAM 帧对齐）
        - p_semantic 来自"目标视频 vs episode 内每个标签的语义阶段序列"的 OTAM 对齐，
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

            # 优先使用预计算的目标特征，避免重复调用 get_feats（backbone 前向）
            if cached_target_features is not None:
                target_features = cached_target_features
            else:
                _, target_features, _ = self.get_feats(
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

            # ===== 向量化（与 _compute_semantic_alignment_loss_multi_class 中 Problem3 修复思路一致）=====
            # 原实现：target_bs × num_classes = 50 次串行 OTAM（10目标视频 × 5类别）
            # 优化后：num_classes = 5 次批量 OTAM，每次同时处理全部 target_bs 个视频
            num_frames = target_features.shape[1]
            feature_dim = target_features.shape[2]
            # 将目标视频特征展平：[target_bs, num_frames, D] → [target_bs*num_frames, D]
            frames_flat = target_features.reshape(target_bs * num_frames, feature_dim)

            semantic_cum_list: List[torch.Tensor] = []
            for cj, cname in enumerate(episode_class_names):
                stage_text_features = episode_stage_feats[cname]  # [num_stages, D]
                # 批量余弦相似度：[target_bs*num_frames, D] × [num_stages, D] → [target_bs*num_frames, num_stages]
                sim_flat = cos_sim(frames_flat, stage_text_features) / self.semantic_temperature
                # 还原为 [target_bs, num_frames, num_stages]，与原循环中单视频的 [num_frames, num_stages] 数值等价
                similarity_matrix = sim_flat.reshape(target_bs, num_frames, -1)
                dists = 1 - torch.sigmoid(similarity_matrix)        # [target_bs, num_frames, num_stages]
                # SEMANTIC_TRANSPOSE=False: 转置为[target_bs,1,num_stages,num_frames]，行=阶段，一阶段对多帧
                # SEMANTIC_TRANSPOSE=True:  原始[target_bs,1,num_frames,num_stages]，行=帧，一帧对多阶段
                if not self.semantic_transpose:
                    dists_4d = dists.transpose(-1, -2).unsqueeze(1)  # [target_bs, 1, num_stages, num_frames]
                else:
                    dists_4d = dists.unsqueeze(1)                     # [target_bs, 1, num_frames, num_stages]
                cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)    # [target_bs, 1]
                semantic_cum_list.append(cum_dists.squeeze(1))       # [target_bs]

            semantic_cum = torch.stack(semantic_cum_list, dim=1)  # [target_bs, num_classes]

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

        优化说明：整个 forward 只调用一次 get_feats（backbone 前向）。
        - 训练时：从原来的 2 次减少到 1 次，OTAM 路径与语义对齐路径共享同一份原始特征。
        - 评估时：从原来的 3 次减少到 2 次，子类预计算一次特征供语义对齐和融合复用。
        """
        support_images, support_labels, target_images, support_real_class = \
            inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']

        if self.training:
            # ===== 训练模式：单次 backbone 前向，所有路径共享原始特征 =====
            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            # ----- 路径1：OTAM 分类（逻辑与父类 CNN_OTAM_CLIPFSAR.forward 训练分支完全一致）-----
            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale
            else:
                class_text_logits = None

            context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)

            # target 经 context2 增强（OTAM 路径与语义对齐路径共享此结果，只计算一次）
            target_features_ctx = self.context2(target_features_raw, target_features_raw, target_features_raw)
            context_support = self.mid_layer(context_support)

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_merged = torch.stack([
                    torch.mean(torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels])
                ctx_merged = torch.stack([
                    torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels])
                # OTAM 用支持集：含文本上下文
                support_features_otam = self.context2(
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1)
                )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                # 语义对齐用支持集：纯视觉，不含文本上下文
                support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                support_features_for_semantic = torch.stack([
                    support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                    for i in range(support_bs)])
            else:
                # OTAM 用支持集：含文本上下文
                support_features_otam = self.context2(
                    torch.cat([support_features_raw, context_support], dim=1),
                    torch.cat([support_features_raw, context_support], dim=1),
                    torch.cat([support_features_raw, context_support], dim=1)
                )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                unique_labels = torch.unique(support_labels)
                support_features_otam = torch.stack([
                    torch.mean(torch.index_select(support_features_otam, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels])
                # 语义对齐用支持集：纯视觉，不含文本上下文
                support_features_for_semantic = self.context2(
                    support_features_raw, support_features_raw, support_features_raw)

            unique_labels = torch.unique(support_labels)
            n_queries = target_features_ctx.shape[0]
            n_support = support_features_otam.shape[0]

            support_flat = rearrange(support_features_otam, 'b s d -> (b s) d')
            target_flat = rearrange(target_features_ctx, 'b s d -> (b s) d')

            frame_sim = cos_sim(target_flat, support_flat)
            frame_dists = 1 - frame_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(
                    rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

            class_dists = torch.stack([
                torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1)
                for c in unique_labels])
            class_dists = rearrange(class_dists, 'c q -> q c')
            model_dict = {'logits': -class_dists, 'class_logits': class_text_logits}

            # ----- 路径2：语义对齐损失（target 特征复用路径1的 target_features_ctx，只计算一次）-----
            try:
                all_features = torch.cat([support_features_for_semantic, target_features_ctx], dim=0)

                all_class_names = []
                for i in range(support_bs):
                    all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        all_class_names.append(self._get_class_name(int(target_real_labels[i].item())))
                else:
                    for i in range(min(target_bs, support_bs)):
                        all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if target_bs > len(all_class_names) - support_bs:
                    last_class = all_class_names[-1] if all_class_names else (
                        self.class_real_train[0] if self.class_real_train else "")
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for _ in range(needed):
                        all_class_names.append(last_class)

                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                debug_semantic = self._debug_count < 3
                if debug_semantic:
                    self._debug_count += 1

                model_dict['semantic_alignment_loss'] = self._compute_semantic_alignment_loss_multi_class(
                    all_features, all_class_names, debug=debug_semantic)
            except Exception as e:
                print(f"Warning: Failed to compute semantic alignment loss: {e}")
                import traceback
                traceback.print_exc()
                model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)

        else:
            # ===== 评估模式 =====
            use_eval_text = hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT
            use_combine = hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE

            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            if use_eval_text or use_combine:
                # 非默认评估路径（EVAL_TEXT / COMBINE）：使用父类 forward 计算 logits
                # get_feats 共调用 2 次（此处 1 次 + super 内部 1 次）
                model_dict = super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).forward(inputs)
                # target context2（供语义对齐使用）
                target_features_ctx = self.context2(
                    target_features_raw, target_features_raw, target_features_raw)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_merged = torch.stack([
                        torch.mean(torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    support_enhanced = self.context2(support_merged, support_merged, support_merged)
                    support_features_for_semantic = torch.stack([
                        support_enhanced[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                else:
                    support_enhanced = self.context2(
                        support_features_raw, support_features_raw, support_features_raw)
                    unique_labels = torch.unique(support_labels)
                    support_enhanced_cls = torch.stack([
                        torch.mean(torch.index_select(support_enhanced, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    support_features_for_semantic = torch.stack([
                        support_enhanced_cls[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
            else:
                # ===== 默认评估路径：内联父类 OTAM 逻辑，context2(target) 只计算一次 =====
                # get_feats 只调用 1 次（上方），不再调用 super().forward()
                # 注意：父类默认评估路径中 context_support 不经过 mid_layer（与训练路径不同）
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale

                context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)

                # target context2：OTAM 路径与语义对齐路径共享，只计算一次（问题4核心优化）
                target_features_ctx = self.context2(
                    target_features_raw, target_features_raw, target_features_raw)

                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_merged = torch.stack([
                        torch.mean(torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    ctx_merged = torch.stack([
                        torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    support_features_otam = self.context2(
                        torch.cat([support_merged, ctx_merged], dim=1),
                        torch.cat([support_merged, ctx_merged], dim=1),
                        torch.cat([support_merged, ctx_merged], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                    # 语义对齐用支持集：纯视觉（不含文本上下文）
                    support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                    support_features_for_semantic = torch.stack([
                        support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                else:
                    support_features_otam = self.context2(
                        torch.cat([support_features_raw, context_support], dim=1),
                        torch.cat([support_features_raw, context_support], dim=1),
                        torch.cat([support_features_raw, context_support], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                    unique_labels = torch.unique(support_labels)
                    support_features_otam = torch.stack([
                        torch.mean(torch.index_select(support_features_otam, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    # 语义对齐用支持集：纯视觉（经过 context2 增强后按类平均，与原始评估保持一致）
                    support_enhanced = self.context2(
                        support_features_raw, support_features_raw, support_features_raw)
                    support_enhanced_cls = torch.stack([
                        torch.mean(torch.index_select(support_enhanced, 0, extract_class_indices(support_labels, c)), dim=0)
                        for c in unique_labels])
                    support_features_for_semantic = torch.stack([
                        support_enhanced_cls[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])

                # OTAM 计算
                unique_labels = torch.unique(support_labels)
                n_queries = target_features_ctx.shape[0]
                n_support = support_features_otam.shape[0]

                support_flat = rearrange(support_features_otam, 'b s d -> (b s) d')
                target_flat = rearrange(target_features_ctx, 'b s d -> (b s) d')
                frame_sim = cos_sim(target_flat, support_flat)
                frame_dists = 1 - frame_sim
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(
                        rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

                class_dists = torch.stack([
                    torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1)
                    for c in unique_labels])
                class_dists = rearrange(class_dists, 'c q -> q c')
                model_dict = {'logits': -class_dists, 'class_logits': class_text_logits}

            # ----- 路径2：语义对齐损失（target 复用 target_features_ctx，不重复调用 context2）-----
            try:
                all_features = torch.cat([support_features_for_semantic, target_features_ctx], dim=0)

                all_class_names = []
                for i in range(support_bs):
                    all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        all_class_names.append(self._get_class_name(int(target_real_labels[i].item())))
                else:
                    for i in range(min(target_bs, support_bs)):
                        all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if target_bs > len(all_class_names) - support_bs:
                    last_class = all_class_names[-1] if all_class_names else (
                        self.class_real_test[0] if self.class_real_test else "")
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for _ in range(needed):
                        all_class_names.append(last_class)

                with torch.no_grad():
                    semantic_alignment_loss = self._compute_semantic_alignment_loss_multi_class(
                        all_features, all_class_names, debug=False)
                model_dict['semantic_alignment_loss'] = semantic_alignment_loss
            except Exception as e:
                print(f"Warning: Failed to compute semantic alignment loss: {e}")
                import traceback
                traceback.print_exc()
                model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)

            # 语义+视觉概率融合，传入已计算的 target_features_raw 避免重复 get_feats
            model_dict = self._fuse_semantic_and_visual_probs_eval(
                inputs, model_dict, cached_target_features=target_features_raw)

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
