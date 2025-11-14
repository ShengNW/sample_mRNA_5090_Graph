# Fig1：概念与流水线（示意）
建议画三个层次：
1) **设计空间**：5'UTR / CDS / 3'UTR 与“目标组织特征（tRNA、RBP、表达背景）”。
2) **模型流水线**：Predictor(Phase1 / `src/side`) → Generator(Phase2 / `src/gen`) → 规则筛选（MFE、RBP位点等）。
3) **命中空间**：Top-k 生成序列进入实验/进一步筛选。

导出：`figs/outputs/fig1_overview.svg` `figs/outputs/fig1_overview.png`
