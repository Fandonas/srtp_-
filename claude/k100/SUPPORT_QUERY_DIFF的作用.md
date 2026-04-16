SUPPORT_QUERY_DIFF 参数作用分析                                                                                                                       
                                                                                                                                                        
  核心区别                                                                                                                                              
                                                                                                                                                        
  两个方法的唯一区别在第 431 行 vs 510 行：                                                                                                             
                  
  - get_seq()（第431行）：
    data["video"] = self.transform(data["video"])
  - get_seq_query()（第510行）：
    data["video"] = self.transform_query(data["video"])

  实际作用

  当 SUPPORT_QUERY_DIFF: true 时：

  1. Support 样本（第242-245行）：

    - 如果 SUPPORT_QUERY_DIFF_SUPPORT: true → 使用 get_seq_query()（transform_query）
    - 否则 → 使用 get_seq()（transform）
  2. Query/Target 样本（第252-255行）：

    - 如果 SUPPORT_QUERY_DIFF: true → 使用 get_seq_query()（transform_query）
    - 否则 → 使用 get_seq()（transform）

  transform vs transform_query 的区别

  这两个 transform 通常有不同的数据增强策略：

  - transform：标准增强（可能包括较弱的增强）
  - transform_query：Query 专用增强（通常包括更强的增强，如随机裁剪、颜色抖动等）

  结论

  ✅ SUPPORT_QUERY_DIFF: true 在 kinetics100 配置中是有效的

  作用：
  - 为 Query/Target 样本应用不同的数据增强策略（transform_query）
  - 为 Support 样本应用标准增强策略（transform）
  - 这是 few-shot 学习中的常见做法，可以提高模型泛化能力

  建议：保留此参数，它能让模型在训练时对 query 样本使用更强的增强，有助于提升性能。