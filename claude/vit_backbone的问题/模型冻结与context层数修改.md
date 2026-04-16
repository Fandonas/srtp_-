---
  修改摘要                                                                                                                                                                                 
                                                                                                                                                                                           
  1. models/base/few_shot.py（第 2719-2725 行新增）                                                                                                                                        
                                                                                                                                                                                         

  在 if/elif backbone 加载块之后、with torch.no_grad(): 之前插入了冻结逻辑：
  # 可选：冻结视觉 backbone，防止 ViT 灾难性遗忘
  if hasattr(cfg.TRAIN, "FREEZE_BACKBONE") and cfg.TRAIN.FREEZE_BACKBONE:
      for param in self.backbone.parameters():
          param.requires_grad = False
      logger.info("Visual backbone FROZEN (TRAIN.FREEZE_BACKBONE=True)")
  else:
      logger.info("Visual backbone trainable (TRAIN.FREEZE_BACKBONE=False/unset)")
  - 位置在 if/elif 块外（8格缩进），对 RN50 和 ViT-B/16 两种 backbone 均生效
  - 使用 hasattr 保护，旧 YAML 不加该字段也不会报错

  2. loss_p_vit.yaml（第 56-57 行新增）

  TRANSFORMER_DEPTH: 1    # context2 层数，默认1；增大可提升表达力但需更多显存
  FREEZE_BACKBONE: true   # 冻结 ViT-B/16 backbone，防止灾难性遗忘（ViT 强烈推荐 true）

  3. loss_p_rn50.yaml（第 56-57 行新增）

  TRANSFORMER_DEPTH: 1    # context2 层数，默认1
  FREEZE_BACKBONE: false  # RN50 backbone 可训练（特征较弱，fine-tune 有益）

---
  验证方法

  训练启动后，日志中应看到：
  - ViT 配置：Visual backbone FROZEN (TRAIN.FREEZE_BACKBONE=True)
  - RN50 配置：Visual backbone trainable (TRAIN.FREEZE_BACKBONE=False/unset)

  如果要测试多层 context2，可将 TRANSFORMER_DEPTH 改为 2，Transformer_v1 会以 depth=2 初始化（现有代码已支持，无需再改 few_shot.py）。