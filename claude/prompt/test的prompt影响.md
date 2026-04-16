---
  USE_CLASSIFICATION=False 时的影响分析                                                                                                                
                                                                                                                                                       
  一、训练阶段                                                                                                                                         
                                                                                                                                                       
  text_features_train —— 间接有效，通过 query_embed_train 起作用

  # __init__ 第 56-60 行
  num_train_classes = self.text_features_train.shape[0]
  self.query_embed_train = nn.Embedding(num_train_classes, self.mid_dim)
  self.query_embed_train.weight = nn.Parameter(
      self.text_features_train.detach().clone()   # 用冻结的文本特征初始化
  )

  # forward 训练分支 第 538 行（USE_CLASSIFICATION=False 仍然执行）
  context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)

  关键区别：

![image-20260325173716945](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20260325173716945.png)

 所以训练阶段 text_features_train 本身不参与前向传播，但它的值决定了 query_embed_train 的初始点，进而影响训练收敛。PROMPT 仍然影响初始化质量。

  class_text_logits —— 训练时为 None，loss 完全不用

  # 第 531-536 行
  if USE_CLASSIFICATION:
      class_text_logits = cos_sim(..., self.text_features_train) * self.scale
  else:
      class_text_logits = None   # ← 直接跳过

  # loss 函数（第 3029 行，继承父类）
  return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
  # 只用 logits，class_logits 完全被忽略

---
  二、评估阶段（默认路径）

  text_features_test —— 直接参与 OTAM，有实质影响

  # 第 680 行（无论 USE_CLASSIFICATION 为何值，默认评估路径都执行）
  context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)
  # [5, 1, 512] — 作为文本语义帧拼入 support

  这个 context_support 直接被送入 context2 注意力：

  support_features_otam = self.context2(
      cat([support_features_raw, context_support], dim=1),  # [5, 9, 512]
      ...
  )[:, :8, :]  # 截取前8帧

  text_features_test 的质量直接影响 support 的帧特征增强效果，进而影响 OTAM 对齐结果。

  评估路径中的 class_text_logits —— 计算了但不影响 loss

  # 第 676-678 行（默认评估路径，每次都计算，不受 USE_CLASSIFICATION 控制）
  feature_classification = self.classification_layer(...).mean(1)
  class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale

  注意这里没有 USE_CLASSIFICATION 判断，评估路径无条件计算，但最终 loss() 只用 logits，这里的计算纯属浪费，不影响结果。

---
  三、汇总

![image-20260325173734945](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20260325173734945.png)