修改总结

  1. 模型代码修改（semantic_alignment_few_shot.py）

  添加了 mid_layer 控制参数（第50-51行）：
  self.use_mid_layer_train = getattr(cfg.TRAIN, 'USE_MID_LAYER_TRAIN', True)
  self.use_mid_layer_eval = getattr(cfg.TRAIN, 'USE_MID_LAYER_EVAL', False)

  训练模式（第565-567行）：
  if self.use_mid_layer_train:
      context_support = self.mid_layer(context_support)

  评估模式（第708-710行）：
  if self.use_mid_layer_eval:
      context_support = self.mid_layer(context_support)

  2. YAML 配置文件（loss_p_rn50.yaml）

  所有参数都设置为 true：
  - USE_QUERY_EMBED_TRAIN: true - 训练时使用可学习的 query embedding
  - USE_QUERY_EMBED_EVAL: true - 评估时也使用 query embedding
  - USE_MID_LAYER_TRAIN: true - 训练时使用 mid_layer
  - USE_MID_LAYER_EVAL: true - 评估时也使用 mid_layer

  这样配置后，训练和评估模式都会使用可学习的 query embedding 并经过 mid_layer 变换，保持一致的特征处理流程。