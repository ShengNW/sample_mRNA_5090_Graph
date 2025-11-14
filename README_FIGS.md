# 三图包（组织特异性表达 mRNA）
本目录为你现有仓库的**增量**结构，用于生成 Fig1–Fig3。把整个 `figs_package_skeleton/` 合并进你的仓库根目录即可。

## 目录
```
data/
  raw/        # 你的原始导出：predict_eval.csv、generated_topk.csv 等
  derived/    # 计算后的中间文件：mfe.csv、rbp_hits.csv 等
scripts/
  calc_mfe.py
  scan_rbp_fimo.py
  eval_predictor.py
  plot_fig2_predictor.py
  plot_fig3_generation.py
  utils.py
figs/
  fig1_overview_guideline.md
  outputs/    # 生成的图片将保存在这里
env/
  requirements.txt
Makefile
```

## 生产顺序
1. 准备两张表：
   - `data/raw/predict_eval.csv`：列至少包含  
     `seq_id, utr5, utr3, y_true, y_pred, organ_id, gc, novelty`
   - `data/raw/generated_topk.csv`：列至少包含  
     `seq_id, utr5, utr3, organ_id_target, score_pred, gc, novelty`
2. 运行结构能量：`make mfe`
3. 运行 RBP 扫描（可选）：`make rbp`
4. 评估预测器并画 Fig2：`make fig2`
5. 生成组学性质与新颖性图（Fig3）：`make fig3`

> Fig1 为概念与流水线示意，建议使用矢量工具（draw.io/PowerPoint）按 `figs/fig1_overview_guideline.md` 绘制。

## 注意
- `calc_mfe.py` 依赖本机可执行的 `RNAfold`（ViennaRNA）。
- `scan_rbp_fimo.py` 依赖 MEME Suite 的 `fimo` 命令及一个 PWM/MEME motif 集合。
- 所有绘图使用 matplotlib，默认保存为 `figs/outputs/*.png` 与 `*.svg`。
