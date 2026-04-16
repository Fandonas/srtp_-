  META_BATCH 的作用

  META_BATCH 是一个布尔值开关，用于控制数据加载模式，决定是使用 Few-shot Learning（元学习） 还是 传统监督学习 模式。

---
  📊 两种模式对比

| 特性         | META_BATCH = True（Few-shot模式）       | META_BATCH = False（传统模式） |
| ------------ | --------------------------------------- | ------------------------------ |
| 数据加载方式 | 每次返回一个完整的episode               | 每次返回单个视频样本           |
| 返回内容     | support set + query set（包含多个样本） | 单个视频 + 标签                |
| 类别选择     | 随机采样 WAY 个类别                     | 固定使用数据集中的类别         |
| 数据集长度   | NUM_SAMPLES（例如 1,000,000）           | 实际视频数量                   |
| 适用场景     | Few-shot学习/元学习                     | 全监督学习                     |

---
  🔍 META_BATCH = True 时的详细流程

  代码位置：datasets/base/ssv2_few_shot.py:203-286

  if self.cfg.TRAIN.META_BATCH:
      # 1. 随机选择 WAY 个类别
      classes = c.get_unique_classes()
      batch_classes = random.sample(classes, self.cfg.TRAIN.WAY)  # 例如选5个类别

      # 2. 对每个类别采样 SHOT + QUERY_PER_CLASS 个样本
      for bc in batch_classes:
          idxs = random.sample(range(n_total), SHOT + n_queries)
    
          # 3. 前 SHOT 个作为 support set
          for idx in idxs[0:SHOT]:
              vid, vid_id = self.get_seq(bc, idx)
              support_set.append(vid)
              support_labels.append(bl)
    
          # 4. 后 QUERY_PER_CLASS 个作为 query set
          for idx in idxs[SHOT:]:
              vid, vid_id = self.get_seq(bc, idx)
              target_set.append(vid)
              target_labels.append(bl)
    
      # 5. 返回一个完整的 episode
      return {
          "support_set": support_set,      # [WAY*SHOT, T, C, H, W]
          "support_labels": support_labels, # [WAY*SHOT]
          "target_set": target_set,         # [WAY*QUERY_PER_CLASS, T, C, H, W]
          "target_labels": target_labels    # [WAY*QUERY_PER_CLASS]
      }

  示例（根据你的配置）：
  WAY: 5                  # 每个episode 5个类别
  SHOT: 1                 # 每类1个support样本
  QUERY_PER_CLASS: 2      # 每类2个query样本

  单个 episode 包含：
  - Support set: 5类 × 1样本 = 5个样本
  - Query set: 5类 × 2样本 = 10个样本
  - 总计：15个样本

---
  🔍 META_BATCH = False 时的流程

  代码位置：datasets/base/ssv2_few_shot.py:290-339

  else:
      # 1. 使用 index 直接索引
      sample_info = self._get_sample_info(index)

      # 2. 解码单个视频
      data, file_to_remove, success = self.decode(sample_info, index)
    
      # 3. 返回单个样本
      return {
          "video": data["video"],  # [T, C, H, W]
          "label": sample_info["supervised_label"]
      }

  这是传统的监督学习模式，类似于 ImageNet 分类。

---
  📏 数据集长度的影响

  代码位置：datasets/base/ssv2_few_shot.py:523-528

  def __len__(self):
      if self.cfg.TRAIN.META_BATCH:
          return self.cfg.TRAIN.NUM_SAMPLES  # 返回 1,000,000
      else:
          return len(self.split_few_shot)     # 返回实际视频数

  在你的配置中：
  META_BATCH: true
  NUM_SAMPLES: 1000000    # 训练时会生成 1,000,000 个随机 episode
  NUM_TRAIN_TASKS: 7000   # 每个 epoch 7000 个 episode

---
  ⚙️ 为什么需要 META_BATCH？

  Few-shot Learning 的核心思想是：
  1. 模型不是学习固定的类别，而是学习"如何快速适应新类别"
  2. 每个 episode 模拟一个新的分类任务（随机选择类别）
  3. 模型需要根据少量 support 样本对 query 样本进行分类

  META_BATCH=True 确保每次迭代都是一个独立的 few-shot 任务，这是元学习的关键。

---
  🎯 总结

  在你的配置文件 configs/projects/CLIPFSAR/hmdb51/4frame_loss_p_vit.yaml:23 中：

  META_BATCH: true  # meta or not

  这行配置的作用是：
  - ✅ 启用 Few-shot Learning 模式
  - ✅ 每次 __getitem__ 返回一个完整的 episode（5-way 1-shot）
  - ✅ 支持元学习训练范式
  - ✅ 数据集会生成 1,000,000 个随机 episode

  如果设置为 META_BATCH: false：
  - ❌ 退化为传统监督学习
  - ❌ 无法进行 Few-shot Learning
  - ❌ 模型将学习固定类别而非元学习能力

  所以对于你的 Few-shot 动作识别任务，必须保持 META_BATCH: true！