● 修改总结                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                  
  修改文件: models/base/semantic_alignment_few_shot.py                                                                                                                                                                                              
                                                                                                                                                                                                                                                  
  核心思路: 将 MSQNet 的可学习标签嵌入机制引入 CLIP-FSAR 小样本学习框架。                                                                                                                                                                           
                                                                                                                                                                                                                                                    
---
  改动 1：__init__ 方法（第 53–61 行）

  新增两个可学习组件：
  - self.text_features_train 从冻结张量转为 nn.Parameter，用 CLIP 文本编码结果初始化，训练时可通过梯度更新
  - self.label_text_projection：一个 nn.Linear(mid_dim, mid_dim, bias=False) 共享投影层，权重用单位矩阵初始化，训练初期等价于恒等映射，不破坏 CLIP 预训练语义

  改动 2：_create_stage_text_features 方法（第 147–187 行）

  将原来整体在 torch.no_grad() 中的逻辑拆为两步：
  - 第一步（no_grad 内）：用冻结的 CLIP 编码器提取各语义阶段的原始文本特征
  - 第二步（no_grad 外）：将原始特征通过 self.label_text_projection 投影，使梯度能流回投影层参与训练

---
  影响分析：
  - 训练路径：text_features_train 变为可学习参数，forward 中索引取值自动带梯度；语义阶段特征经投影层后也有梯度
  - 测试路径：text_features_test 保持冻结不变，但语义阶段特征仍经过训练好的投影层，实现对未见类别的泛化
  - 父类兼容：nn.Parameter 是 Tensor 子类，父类中的索引操作不受影响，无需改动其他文件