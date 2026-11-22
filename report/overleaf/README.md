# Overleaf 版报告（基于深大官方模板）

来源模板：[ChaoFan996/Experiment-Report-Template-SZU](https://github.com/ChaoFan996/Experiment-Report-Template-SZU)
（Chao Fan / Kewei Ou，人工智能学院）。本目录保留了模板的封面字段、章节结构与末尾批阅页，只把正文替换成 Project 1 的实际内容。

## 在 Overleaf 上使用

1. 打开 <https://www.overleaf.com/> → New Project → **Upload Project**，上传本目录内全部文件（或打包 zip）。
2. 必须包含：`main.tex` 和 `figs/vit_overview.png`（缺图会编译报错）。
3. Menu → **Compiler 选 XeLaTeX**（模板用 `xeCJK` + `Noto Serif CJK SC`，Overleaf 自带该字体）。
4. 点 Recompile，下载生成的 PDF。

## 提交前需要你改的两处

- 封面 `学号`、`提交时间`：我没有这些信息，留空了，请填上。
- 封面 `GitHub: <在此填入仓库链接>`：换成你的仓库地址。
- 指导教师一栏沿用模板原值「舒婷」，如与实际情况不符请自行修改。

## 页数说明（重要）

课程要求**正文少于 5 页（封面不计）**。模板最后一页是「批阅与成绩评定页」，是给老师填的，通常不计入正文；但如果你担心页数，有两个保底选择：

1. 用仓库里的 `report/实验报告.pdf` —— 已实测 **封面 + 3 页正文 = 4 页**，确定达标，且已含插图、红色高亮、误差表、形状变换、bug 与声明；
2. 在 Overleaf 编译后若正文偏长，删除 `main.tex` 最后一段「批阅与成绩评定页」（从 `\newpage` 后的注释到 `\end{document}` 之前），或把 `\geometry` 的边距再调小 0.05in。

本机没有 LaTeX 发行版，所以 Overleaf 版**未在本地编译验证页数**；环境配对与花括号平衡已做静态检查。

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` | 基于官方模板改写的主文件（XeLaTeX） |
| `figs/vit_overview.png` | 架构插图，由 `report/figures/architecture.svg` 渲染而来 |
| `README.md` | 本说明 |
