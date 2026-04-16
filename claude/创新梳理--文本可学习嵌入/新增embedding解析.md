---
  新增 Embedding 流程详解                                                                                                                                                                               
                                                                                                                                                                                                        
  涉及的两个组件                                                                                                                                                                                        
                                                                                                                                                                                                        
  text_features_train  (nn.Parameter)     ← 训练集所有类别的 CLIP 标签嵌入，可学习
  label_text_projection (nn.Linear)       ← 共享投影层，将 CLIP 文本空间映射到任务适配空间

---
  第一步：__init__（第 55-60 行）

  # 父类原始值：backbone.encode_text(text_templete) → [num_train_classes, mid_dim] 普通 tensor
  # 子类做法：包装成 nn.Parameter，变为可学习参数
  self.text_features_train = nn.Parameter(self.text_features_train.clone().float())

  # 投影层：mid_dim × mid_dim，无偏置，单位矩阵初始化
  self.label_text_projection = nn.Linear(self.mid_dim, self.mid_dim, bias=False)
  nn.init.eye_(self.label_text_projection.weight)  # 初期等价于恒等映射

  以 ViT-B/16 为例：mid_dim=512，所以 label_text_projection.weight 是 512×512 的单位矩阵。

---
  第二步：_create_stage_text_features（第 157-185 行，仅执行一次）

  # === 第一步（no_grad 内）：CLIP 编码器冻结，提取原始语义阶段特征 ===
  raw_stage_features = {}
  with torch.no_grad():
      for class_name, stages in self._semantic_stages.items():
          #   例：class_name = "pick up ball then throw"
          #   stages = ["pick up ball", "throw"]
          stage_features = []
          for stage in stages:
              stage_text = f"a video of {stage}"    # "a video of pick up ball"
              text_tokens = tokenize([stage_text]).to(device)
              stage_feature = self._full_clip_model.encode_text(text_tokens)  # [1, 512]
              stage_features.append(stage_feature)

          raw_stage_features[class_name] = torch.cat(stage_features, dim=0)  # [2, 512]

  # === 第二步（no_grad 外）：通过投影层，梯度可流回 label_text_projection ===
  for class_name, raw_features in raw_stage_features.items():
      stage_text_features[class_name] = self.label_text_projection(raw_features.float())
      # 计算图：raw_features（无grad）→ weight（有grad）→ 输出（有grad）
      # 结果：[2, 512]，可对 label_text_projection.weight 产生梯度

  具体例子（5-way 1-shot，mid_dim=512）：
  - raw_features: [2, 512] - 两个阶段的原始 CLIP 特征，无梯度
  - 投影后 _stage_text_features["pick up ball then throw"]: [2, 512]

---
  第三步：forward 训练分支（第 522-525 行）

  # context2 是父类的 Temporal Transformer (MultiHeadAttention)
  target_features_enhanced = self.context2(target_features, target_features, target_features)

  # === 新增的两行 ===
  context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # [support_bs, 1, 512]
  context_support = self.mid_layer(context_support)  # [support_bs, 1, 512]

---
  逻辑正确性检查

  问题 1（严重）：label_text_projection 的训练更新实际失效

  第 1 次 forward：
    _text_features_initialized = False
    → 调用 _create_stage_text_features()
    → label_text_projection（单位矩阵）处理 raw_features
    → 结果缓存到 _stage_text_features
    → _text_features_initialized = True  ← 锁死！

  第 1 次 backward：
    → label_text_projection.weight 被更新（权重已不再是单位矩阵）

  第 2 次 forward：
    _text_features_initialized = True
    → 直接跳过！_stage_text_features 仍是第 1 次的缓存值
    → label_text_projection 的最新权重对语义阶段文本特征毫无影响

  实际效果：_stage_text_features 在整个训练过程中等价于 用初始单位矩阵投影一次后完全冻结，label_text_projection 的训练作用对语义阶段文本特征为零。

---
  问题 2（中等）：context_support 是未完成的死代码

  # 第 524-525 行：
  context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)
  context_support = self.mid_layer(context_support)
  # ↑ 计算完成后... 下面第 528 行直接进入 MERGE_BEFORE 分支，context_support 从未被使用！

  注释 #改加入文本qery 说明原本打算将 context_support 作为 query 传入 context2（模仿父类第 2827 行的 cat 操作），但代码没有完成这个改动。这两行完全是死代码。

---
  正确的部分：text_features_train 作为 nn.Parameter

  虽然子类自己的 context_support 是死代码，但 text_features_train 通过 父类 super().forward(inputs) 仍然正常参与训练：

  # 父类 few_shot.py 第 2805 行（USE_CLASSIFICATION 分支）：
  class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale
  # ↑ 有梯度路径，text_features_train 会被更新 ✓

  # 父类 few_shot.py 第 2811 行：
  context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)
  support_features = torch.cat([support_features, context_support], dim=1)
  support_features = self.context2(support_features, ...)
  # ↑ 有梯度路径，text_features_train 会被更新 ✓

  所以 text_features_train → nn.Parameter 的设计逻辑正确，它通过父类 forward 参与训练。

---
  总结

  ┌─────────────────────────────────────────────────┬──────────────────────────────┬──────────────────────────────────┬────────────────────────────────────┐
  │                      组件                       │           设计意图           │             实际效果             │                问题                │
  ├─────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────┤
  │ text_features_train → nn.Parameter              │ 类标签嵌入可学习             │ 正确，通过父类 forward 参与训练  │ 无                                 │
  ├─────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────┤
  │ label_text_projection 投影 _stage_text_features │ 语义阶段文本特征随训练自适应 │ 失效，只在第一次初始化时运行一次 │ 缓存机制导致权重更新对文本特征无效 │
  ├─────────────────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────┤
  │ 子类 forward 中 context_support 两行            │ 将文本特征融入 support       │ 死代码，从未被使用               │ 未完成的功能                       │
  └─────────────────────────────────────────────────┴──────────────────────────────┴──────────────────────────────────┴────────────────────────────────────┘