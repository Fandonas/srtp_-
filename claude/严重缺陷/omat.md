  唯一的不完美

  转置后，列索引1对应帧1（而不是帧0），OTAM
  在列1处保留了垂直项，导致帧1仍可被多个阶段匹配。这是边界条件的副作用，属于轻微瑕疵，其他所有帧均满足"一帧只对一个阶段"的约束。

---
  代码层面的改动位置

  只需在两个函数中各加一行 .transpose(-1, -2)：

  _compute_semantic_alignment_loss（第243行附近）：
  # 现在：dists.unsqueeze(0).unsqueeze(0)  → [1,1,T,S]
  # 改为：dists.T.unsqueeze(0).unsqueeze(0) → [1,1,S,T]

  _compute_semantic_alignment_loss_multi_class（第350行附近）：
  # 现在：dists.unsqueeze(1)              → [B,1,T,S]
  # 改为：dists.transpose(-1,-2).unsqueeze(1) → [B,1,S,T]

  不需要动 OTAM_cum_dist_v2 本身，也不影响帧对齐路径。