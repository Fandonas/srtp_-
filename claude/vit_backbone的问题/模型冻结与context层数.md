 计划：YAML 配置 context2 层数 + 冻结视觉 Backbone      

 Context

 用户希望在 YAML 配置文件中控制两个行为：
 1. context2（Transformer_v1）的层数
 2. 是否冻结视觉 backbone（CLIP 的 visual encoder）

 这两个参数对 ViT-B/16 训练尤其重要：ViT 因灾难性遗忘导致 min_top1_err 无法提升，冻结 backbone 是标准解法。

---
 发现：TRANSFORMER_DEPTH 已存在

 few_shot.py 第 2749-2752 行已有完整实现：

 if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
     self.context2 = Transformer_v1(..., depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
 else:
     self.context2 = Transformer_v1(...)   # depth 默认=1

 无需修改 few_shot.py，只需在 YAML 里加一行即可生效。

---
 需要新增：FREEZE_BACKBONE

 few_shot.py 目前没有任何 backbone 冻结逻辑。需要在两处 backbone 加载之后、with torch.no_grad(): 之前（第 2718→2719 行之间）插入：

         # 新增：可选冻结视觉 backbone
         if hasattr(cfg.TRAIN, "FREEZE_BACKBONE") and cfg.TRAIN.FREEZE_BACKBONE:
             for param in self.backbone.parameters():
                 param.requires_grad = False
             logger.info("Visual backbone FROZEN (TRAIN.FREEZE_BACKBONE=True)")
         else:
             logger.info("Visual backbone trainable (TRAIN.FREEZE_BACKBONE=False/unset)")

---
 修改文件清单

 1. models/base/few_shot.py（需修改）

 - 位置：第 2718 行（self.mid_dim = 512）和第 2719 行（with torch.no_grad():）之间
 - 内容：插入 FREEZE_BACKBONE 检查代码（见上）

 2. configs/projects/CLIPFSAR/hmdb51/loss_p_vit.yaml（需修改）

 在 TRAIN: 块中添加：
   TRANSFORMER_DEPTH: 1    # context2 层数，默认1；增大可提升表达力但需更多显存
   FREEZE_BACKBONE: true   # 冻结 ViT-B/16 backbone，防止灾难性遗忘（ViT 强烈推荐 true）

 3. configs/projects/CLIPFSAR/hmdb51/loss_p_rn50.yaml（需修改）

 在 TRAIN: 块中添加：
   TRANSFORMER_DEPTH: 1    # context2 层数，默认1
   FREEZE_BACKBONE: false  # RN50 backbone 可训练（特征较弱，fine-tune 有益）

---
 验证方式

 1. 运行训练，观察日志中是否出现：
   - "Visual backbone FROZEN" 或 "Visual backbone trainable"
 2. 检查 ViT 模型的 min_top1_err 是否在后续 epoch 继续下降（不再锁死在 epoch 1）
 3. 可用 TRANSFORMER_DEPTH: 2 测试多层 context2 是否生效（日志无报错）