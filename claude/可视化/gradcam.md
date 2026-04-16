• 已新增脚本 runs/gradcam_few_shot.py。它是独立工具，不改你现有训练/测试流程，支持两种骨干：

  - RN50：hook 在 head.backbone.layer4[-1]
  - ViT-B/16：按之前方案，hook 在 head.backbone.conv1

  脚本会复用项目自己的 Config、build_model、load_test_checkpoint、build_loader(cfg, "test")，然后从一个 few-shot episode 里取一条 query 反传，保存：

  - frame_00.png 等逐帧结果
  - montage.png
  - cam.npy
  - metadata.json

  静态检查我已经做过：

  - python -m py_compile runs\gradcam_few_shot.py 通过

  直接用法示例。

  当前这份 loss_p_rn50.yaml 实际内容还是 ViT-B/16：

  python runs/gradcam_few_shot.py ^
    --cfg configs/projects/CLIPFSAR/hmdb51/loss_p_rn50.yaml ^
    --checkpoint D:\your_ckpt\checkpoint_epoch_00023.pyth

  如果你要强制按 RN50 版可视化：

  python runs/gradcam_few_shot.py ^
    --cfg configs/projects/CLIPFSAR/hmdb51/loss_p_rn50.yaml ^
    --backbone RN50 ^
    --checkpoint D:\your_rn50_ckpt\checkpoint_epoch_00023.pyth

  几个常用参数：

  - --episode_index 0 选第几个 episode
  - --query_index 0 选第几个 query
  - --score_mode pred|gt|index 反传预测类、真值类或指定类
  - --class_index 2 在 score_mode=index 时使用
  - --frame_index 3 只保存某一帧；默认 -1 保存全部帧

  注意两点：

  - 这个脚本需要可用 GPU，因为你这套 few-shot CLIP 头初始化时直接走 CUDA。
  - 我只做了语法检查，还没替你跑完整 Grad-CAM，因为那需要你的真实 checkpoint 和数据环境。

  如果你要，我下一步可以继续帮你补一版“批量处理多个 query/多个 episode”的导出模式。