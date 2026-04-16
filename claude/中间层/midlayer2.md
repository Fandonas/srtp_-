 修改总结
                                                                                                                                                       
  1. 模型代码修改（semantic_alignment_few_shot.py）
                                                                                                                                                       

  添加 mid_layer2 控制参数和层（第55、74-75行）：                                                                                                    
  self.use_mid_layer2 = getattr(cfg.TRAIN, 'USE_MID_LAYER2', False)                                                                                    

  # 创建 mid_layer2：与 mid_layer 并行处理
  if self.use_mid_layer2:
      self.mid_layer2 = nn.Linear(self.mid_dim, self.mid_dim)

  训练模式并行处理（第578-580行）：
  if self.use_mid_layer2:
      context_support2 = self.mid_layer2(context_support)

  评估模式并行处理（第728-730行）：
  if self.use_mid_layer2:
      context_support2 = self.mid_layer2(context_support)

  2. YAML 配置文件（loss_p_rn50.yaml）

  添加了 USE_MID_LAYER2: false 参数，默认不启用。

  特点说明

  - mid_layer2 与 mid_layer 并行处理（不是串行）
  - context_support2 目前只是计算出来，暂不作为模型输入
  - 通过 USE_MID_LAYER2 参数控制是否启用
  - 当需要使用 context_support2 时，可以很方便地将其加入模型

修改总结
                                                                                                                                                                                      
  核心逻辑（以训练 else 分支为例）
                                                                                                                                                                                      
  启用 USE_MID_LAYER2_SEMANTIC_TRAIN=true 后，support_features_for_semantic 的计算从：                                                                                              
  # 纯视觉，无文本上下文                                                                                                                                                              
  support_features_for_semantic = context2(support_raw, support_raw, support_raw)
  变为：
  # context_support2 注入，与 OTAM 强化方式完全一致
  support_features_for_semantic = context2(
      cat([support_raw, context_support2]),
      cat([support_raw, context_support2]),
      cat([support_raw, context_support2])
  )[:, :NUM_INPUT_FRAMES, :]

  新增参数（semantic_alignment_few_shot.py + loss_p_rn50.yaml）

![image-20260329105302527](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20260329105302527.png)