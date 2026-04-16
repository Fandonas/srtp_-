---
  self.context2 是什么

  初始化位置：few_shot.py:2749

  self.context2 = Transformer_v1(
      dim=self.mid_dim,          # 512（ViT-B/16）或 1024（RN50）
      heads=8,
      dim_head_k=self.mid_dim//8, # 每个注意力头的维度 = 64
      dropout_atte=0.2,
      depth=1                    # 默认深度为1，可由 TRANSFORMER_DEPTH 覆盖
  )

  本质上是一个 带 Q/K/V 独立投影的多头交叉注意力 Transformer 块。

---
  内部结构拆解

  Transformer_v1 由两个子模块组成：

  PreNormattention_qkv
    └── LayerNorm(q, k, v) + Attention_qkv + 残差连接
  FeedForward
    └── Linear → GELU → Dropout → Linear → Dropout + 残差连接

  子模块1：PreNormattention_qkv（few_shot.py:974）

  def forward(self, q, k, v):
      return self.fn(self.norm(q), self.norm(k), self.norm(v)) + q
      #              ↑ 先LayerNorm                              ↑ 残差加到q上

  先对三个输入做 LayerNorm，再送入注意力计算，最后以 q 作为残差连接。

---
  子模块2：Attention_qkv（few_shot.py:1038）

  这是多头注意力的核心，以 dim=512, heads=8, dim_head=64 为例：

  输入: q, k, v 形状均为 [bs, T, 512]

  to_q: Linear(512 → 512)  →  q: [bs, T, 512]
  to_k: Linear(512 → 512)  →  k: [bs, T, 512]
  to_v: Linear(512 → 512)  →  v: [bs, T, 512]

  rearrange → q: [bs, 8, T, 64]   （8个头，每头64维）
           → k: [bs, 8, T, 64]
           → v: [bs, 8, T, 64]

  dots = einsum(q, k) * scale     → [bs, 8, T, T]   （注意力权重矩阵）
         ↓ softmax(dim=-1)
  attn                            → [bs, 8, T, T]   （归一化为概率分布）
         ↓ 加权sum(v)
  out = einsum(attn, v)           → [bs, 8, T, 64]
         ↓ rearrange
  out                             → [bs, T, 512]
         ↓ to_out: Linear(512→512) + Dropout
  最终输出                          → [bs, T, 512]   （形状不变）

  scale = 64^(-0.5) = 0.125，防止点积过大导致 softmax 梯度消失。

---
  子模块3：FeedForward（few_shot.py:1646）

  输入: [bs, T, 512]
  ↓ Linear(512 → 512) + GELU + Dropout
  ↓ Linear(512 → 512) + Dropout
  输出: [bs, T, 512]   （形状不变）

  最终: ff(x) + x  →  残差连接

---
  Transformer_v1.forward 整体逻辑（few_shot.py:993）

  def forward(self, q, k, v):
      # 第1层：允许交叉注意力（q, k, v 可以来自不同源）
      for attn, ff in self.layers[:1]:
          x = attn(q, k, v)       # 交叉注意力
          x = ff(x) + x           # FFN + 残差

      # 第2层起：强制自注意力（x作为自己的q,k,v）
      if self.depth > 1:
          for attn, ff in self.layers[1:]:
              x = attn(x, x, x)   # 自注意力
              x = ff(x) + x
      return x

  关键点：第1层支持传入不同的 q/k/v（实现交叉注意力），之后所有层都强制自注意力。

---
  在 semantic_alignment_few_shot.py 中如何使用

  # Target 特征增强（自注意力）
  target_features_enhanced = self.context2(
      target_features, target_features, target_features
  )
  # q=k=v，帧与帧之间互相学习时序关系

  # Support 特征增强（自注意力）
  support_features_enhanced = self.context2(
      support_features, support_features, support_features
  )

  两者都是纯自注意力，每一帧可以「看到」同视频的所有其他帧，从而聚合时序上下文信息。

---
  与父类用法的对比

  父类 CNN_OTAM_CLIPFSAR.forward 在 support 上使用了文本引导的交叉注意力：

  # 父类（few_shot.py:2827）
  context_support = self.text_features_train[support_real_class].unsqueeze(1)  # [bs,1,512]
  support_features = torch.cat([support_features, context_support], dim=1)     # [bs,T+1,512]
  support_features = self.context2(support_features, support_features, support_features)
                     [:, :NUM_INPUT_FRAMES, :]  # 截取前T帧

  父类把文本特征拼接到帧序列末尾，让帧可以"看见"类别语义描述后再做注意力。而子类（语义对齐模型）去掉了这个文本引导，保持纯视觉自注意力。

---
  维度完整流程（以 ViT-B/16，T=8，bs=5 为例）

  输入:  [5, 8, 512]
    ↓ LayerNorm(q/k/v)        → [5, 8, 512]  × 3
    ↓ to_q/k/v Linear         → [5, 8, 512]  × 3
    ↓ rearrange(8头)           → [5, 8, 8, 64] × 3
    ↓ einsum(q,k)*0.125        → [5, 8, 8, 8]   (帧间注意力权重)
    ↓ softmax                  → [5, 8, 8, 8]
    ↓ einsum(attn,v)           → [5, 8, 8, 64]
    ↓ rearrange合并头          → [5, 8, 512]
    ↓ Linear(512→512)+Dropout  → [5, 8, 512]
    ↓ + 残差(q)                → [5, 8, 512]
    ↓ FFN(512→512)+残差        → [5, 8, 512]
  输出: [5, 8, 512]   ← 形状完全不变，但每帧特征已融合了全局时序信息

  注意力矩阵 [5, 8, 8, 8] 的含义是：5个视频，8个注意力头，8帧对8帧的关注程度——每一帧都能关注到同视频所有帧，实现时序信息聚合。