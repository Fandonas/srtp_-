---
  当前状况诊断

  训练日志显示训练已跑完全部 7000 个任务（完整一轮）：

  {"_type": "train_epoch", "epoch": "10/300", "iter": "7000/1000000", ...}

  NUM_TRAIN_TASKS: 7000 + STEPS_ITER: 700 = 10 个 epoch，全部完成。但检查点目录不存在，说明 checkpoint 保存存在问题。

---
  Checkpoint 保存机制说明

  代码每 VAL_FRE_ITER=300 次迭代保存一次，23 次验证意味着 save_checkpoint 被调用了 23 次，文件应保存于：

  output/CLIPFSAR_HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot/checkpoints/
    checkpoint_epoch_00001.pyth  (iter 300)
    checkpoint_epoch_00002.pyth  (iter 600)
    ...
    checkpoint_epoch_00023.pyth  (iter 6900)

  但实际上该目录不存在。全项目范围内也未找到任何 .pyth 文件，说明 checkpoint 保存在 Windows 上存在系统性问题（可能是 Windows 文件写入权限或路径问题）。

---
  继续训练的两种方式

  方式一：自动恢复（推荐，无需改 yaml）

  配置中已有 AUTO_RESUME: true（继承自 finetune.yaml）。只要 checkpoint 文件存在，下次运行相同命令即可自动从最新 checkpoint 恢复：

  python runs/run.py --cfg configs/projects/CLIPFSAR/hmdb51/loss_p_rn50.yaml

  代码逻辑（utils/checkpoint.py:532）：
  if cfg.TRAIN.AUTO_RESUME and has_checkpoint(cfg.OUTPUT_DIR):
      last_checkpoint = get_last_checkpoint(cfg.OUTPUT_DIR)  # 按文件名排序取最新
      checkpoint_epoch = load_checkpoint(...)  # 恢复模型+优化器状态
      start_epoch = checkpoint_epoch + 1       # 从下一个epoch继续

  方式二：手动指定 checkpoint 文件

  在 yaml 中设置（需先有一个 .pyth 文件）：

  TRAIN:
    CHECKPOINT_FILE_PATH: "output/CLIPFSAR_HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot/checkpoints/checkpoint_epoch_00023.pyth"

---
  当前问题的根本原因与修复

  由于此次训练 checkpoint 实际上没有被保存，需要重新训练。可以先验证 checkpoint 是否能正常写入：

  python -c "
  import os
  os.makedirs('output/CLIPFSAR_HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot/checkpoints', exist_ok=True)
  import torch
  torch.save({'test': 1}, 'output/CLIPFSAR_HMDB51_SEMANTIC_ALIGNMENT_FEW_SHOT_1shot/checkpoints/test.pyth')
  print('写入成功')
  "

  如果写入成功，下次训练时 checkpoint 应该能正常保存，然后 AUTO_RESUME: true 就能在中断后自动恢复。