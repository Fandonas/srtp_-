---
  1. configs/projects/CLIPFSAR/hmdb51/loss_p_rn50.yaml                                                                                                       

  新增 1 行（第 34 行）：                                                                                                                                    
  # 改前                                                                                                                                                   
    BATCH_SIZE_PER_TASK: 2
    TEXT_COFF: 0.5

  # 改后
    BATCH_SIZE_PER_TASK: 2
    USE_AMP: true  # 开启自动混合精度（AMP）；OTAM 和语义对齐 sigmoid 路径内部强制 float32，训练精度不受影响
    TEXT_COFF: 0.5

---
  2. runs/train_net_few_shot.py

  改动 1：函数开头初始化 scaler（第 51-52 行）
  # 改前
      model.train()
      norm_train = False

  # 改后
      model.train()
      use_amp = getattr(cfg.TRAIN, 'USE_AMP', False)
      scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
      norm_train = False

  改动 2：前向传播包 autocast（第 105-106 行）
  # 改前
          else:
              model_dict = model(task_dict)

  # 改后
          else:
              with torch.cuda.amp.autocast(enabled=use_amp):
                  model_dict = model(task_dict)

  改动 3：backward/optimizer 换成 scaler 版本（第 148-156 行）
  # 改前
          if torch.isnan(loss).any():
              loss.backward(retain_graph=False)
              optimizer.zero_grad()
              continue
          loss.backward(retain_graph=False)
          if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
              nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)
          if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
              optimizer.step()
              optimizer.zero_grad()

  # 改后
          if torch.isnan(loss).any():
              optimizer.zero_grad()
              continue
          scaler.scale(loss).backward(retain_graph=False)
          if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
              if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
                  scaler.unscale_(optimizer)
                  nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)
              scaler.step(optimizer)
              scaler.update()
              optimizer.zero_grad()

---
  3. models/base/few_shot.py

  改动：OTAM_cum_dist_v2 函数体全部包进双重保险（第 2660-2691 行）
  # 改前
  def OTAM_cum_dist_v2(dists, lbda=0.5):
      """..."""
      dists = F.pad(dists, (1,1), 'constant', 0)
      cum_dists = torch.zeros(dists.shape, device=dists.device)
      for m in range(...):
          ...
      for l in range(...):
          ...log/exp...
      return cum_dists[:,:,-1,-1]

  # 改后
  def OTAM_cum_dist_v2(dists, lbda=0.5):
      """..."""
      with torch.cuda.amp.autocast(enabled=False):
          dists = dists.float()
          dists = F.pad(dists, (1,1), 'constant', 0)
          cum_dists = torch.zeros(dists.shape, device=dists.device)
          for m in range(...):
              ...
          for l in range(...):
              ...log/exp...
          return cum_dists[:,:,-1,-1]

---
  4. models/base/semantic_alignment_few_shot.py

  改动 1：_compute_semantic_alignment_loss_multi_class 的 sigmoid+OTAM 块（第 391-400 行）
  # 改前
                  dists = 1 - torch.sigmoid(similarity_matrix.float())
                  if not self.semantic_transpose:
                      dists_4d = dists.transpose(-1,-2).unsqueeze(1)
                  else:
                      dists_4d = dists.unsqueeze(1)
                  cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)

  # 改后
                  with torch.cuda.amp.autocast(enabled=False):
                      dists = 1 - torch.sigmoid(similarity_matrix.float())
                      if not self.semantic_transpose:
                          dists_4d = dists.transpose(-1,-2).unsqueeze(1)
                      else:
                          dists_4d = dists.unsqueeze(1)
                      cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)

  改动 2：_fuse_semantic_and_visual_probs_eval 的 sigmoid+OTAM 块（第 532-538 行）
  # 改前
                  dists = 1 - torch.sigmoid(similarity_matrix.float())
                  if not self.semantic_transpose:
                      dists_4d = dists.transpose(-1,-2).unsqueeze(1)
                  else:
                      dists_4d = dists.unsqueeze(1)
                  cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)

  # 改后
                  with torch.cuda.amp.autocast(enabled=False):
                      dists = 1 - torch.sigmoid(similarity_matrix.float())
                      if not self.semantic_transpose:
                          dists_4d = dists.transpose(-1,-2).unsqueeze(1)
                      else:
                          dists_4d = dists.unsqueeze(1)
                      cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)

---
  总览

![image-20260402154746307](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20260402154746307.png)