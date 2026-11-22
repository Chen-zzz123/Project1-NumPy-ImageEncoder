# Overleaf 版报告（基于深大官方模板）

来源模板：[ChaoFan996/Experiment-Report-Template-SZU](https://github.com/ChaoFan996/Experiment-Report-Template-SZU)
（Chao Fan / Kewei Ou，人工智能学院）。本目录保留了模板的封面字段、章节结构与末尾批阅页，只把正文替换成 Project 1 的实际内容。

## 在 Overleaf 上使用

1. 打开 <https://www.overleaf.com/> → New Project → **Upload Project**，上传本目录内全部文件（或打包 zip）。
2. 必须包含：`main.tex` 和 `figs/vit_overview.png`（缺图会编译报错）。
3. Menu → **Compiler 选 XeLaTeX**（模板用 `xeCJK` + `Noto Serif CJK SC`，Overleaf 自带该字体）。
4. 点 Recompile，下载生成的 PDF。

## 封面信息（已填好，提交前请核对）

- `报告人`：陈思翰；`学号`：2025681073；`实验时间` / `提交时间`：2026-11-22。
- `GitHub`：https://github.com/Chen-zzz123/Project1-NumPy-ImageEncoder，如换了仓库请同步修改。
- ⚠️ `指导教师` 一栏沿用模板原值「舒婷」，**不确定是否与本课程实际任课教师一致，请务必核对并修改**。

## 页数与编译状态（已实测）

本目录 `main.tex` 已在本地 MiKTeX（XeLaTeX 25.12）**实测编译通过**：

- **总 5 页 = 封面 1 + 正文 3 + 批阅页 1**；课程要求"正文少于 5 页（封面不计）"→ 正文 3 页达标。
- 编译第二遍后无 Overfull/未定义引用等警告。
- 字体说明：模板原设 `Noto Serif CJK SC`（Overleaf 自带）；本机 Windows 编译副本改用 `Noto Serif SC`（本机已装），两份 PDF 视觉一致。
- 已产出成品：`../实验报告-SZU模板版.pdf`（即本模板的编译产物，可直接提交）。

课程要求**正文少于 5 页（封面不计）**。模板最后一页是「批阅与成绩评定页」，是给老师填的，不计入正文；如仍希望去掉，删除 `main.tex` 最后一段「批阅与成绩评定页」（从 `\newpage` 后的注释到 `\end{document}` 之前）即可，正文不变。

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` | 基于官方模板改写的主文件（XeLaTeX） |
| `figs/vit_overview.png` | 架构插图，由 `report/figures/architecture.svg` 渲染而来 |
| `README.md` | 本说明 |
