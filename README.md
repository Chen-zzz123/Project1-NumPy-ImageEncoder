# Project 1 — NumPy 复现 OpenCLIP ViT-B/32 图像编码器

> **One-paragraph summary (EN):** This repository reproduces the *forward pass* of the
> OpenCLIP ViT-B/32 image tower using **NumPy only** (no PyTorch, no training, no network).
> All 152 learned weight tensors are supplied by the course. The implementation passes
> every stage of the provided `check.py`, with max absolute output error `5.44e-07`
> (tolerance `2e-5 + 2e-4·|ref|`). See `report/` for the lab report.

用纯 NumPy 把一张图片变成一行 512 维向量：切图块 → 投影 → 12 层 Transformer → 取 CLS → 归一化。**只改了 `image_encoder.py` 一个文件**，其余都是课程提供的原始文件（未修改）。

---

## 1. 如何复现（Reproduce）

环境：Python 3.9+（实测 3.13.14）、无需 GPU、无需联网（依赖已装好）。

```bash
# 进入项目目录
cd project1-numpy-student

# 安装依赖（numpy / Pillow / imageio）
python -m pip install -r requirements.txt

# 全部 8 个阶段的数值检查（约 10 秒）
python check.py --stage all

# 生成 10 张示例图的输出向量
python run.py --output output_tokens.npz

# 对任意图片生成向量
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`run.py` 输出的 `.npz` 里有两个数组：`ids`（图片名，顺序与输入一致）和
`output_tokens`（`float32 [B,512]`，每行长度为 1）。

**想单独查某一阶段**（出错时从第一个 FAIL 的阶段往回修）：

```bash
python check.py --stage patches      # 切图块
python check.py --stage tokens       # 图块投影 + CLS + 位置编码
python check.py --stage block0       # 第 1 个 Transformer 块
python check.py --stage last_tokens  # 12 个块之后的完整 token 序列
python check.py --stage output       # 最终 512 维输出
python check.py --stage batch        # 打乱顺序 / 重复图片
```

课堂迷你算例（理解注意力算术用）：`python transformer_steps.py`。

## 2. 数值检查结果（实测，2026-11-22）

| Stage | Shape | max abs error | mean abs error | 容差 (atol / rtol) | 结果 |
|---|---|---|---|---|---|
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | 2e-6 / 2e-4 | PASS |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | 2e-4 / 2e-4 | PASS |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | 2e-4 / 2e-4 | PASS |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | 2e-4 / 2e-4 | PASS |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | 2e-4 / 2e-4 | PASS |
| output | (10,512) | **5.439e-07** | 8.311e-08 | 2e-5 / 2e-4 | PASS |
| batch | (3,512) | **4.619e-07** | 7.632e-08 | 2e-5 / 2e-4 | PASS |

原始输出存档：`report/check_results.txt`。参考数组只被 `check.py` 读取，
**从不作为编码器输入**；实现能处理任意图片与任意 batch size。

## 3. 我实现了什么（`image_encoder.py`）

| TODO | 方法 | 关键操作 | 形状 |
|---|---|---|---|
| 01 | `patchify` | `reshape(B,3,7,32,7,32).transpose(0,2,4,1,3,5).reshape(B,49,3072)` | `[B,3,224,224]` → `[B,49,3072]` |
| 02 | `split_heads` | `reshape(B,N,12,64).transpose(0,2,1,3)` | `[B,N,768]` → `[B,12,N,64]` |
| 03 | `MultiHeadAttention.__call__` | `QKᵀ/√64` → softmax → `@V` → 合头 → 输出投影 | `[B,12,50,50]` → `[B,50,768]` |
| 04 | `TransformerBlock.__call__` | 两次残差相加 + `GELU(uW₁ᵀ+b₁)W₂ᵀ+b₂` | `[B,50,768]` 不变 |
| 05 | `embed` | 图块投影（无偏置）+ 前置 CLS + 广播加位置编码 | `[B,49,3072]` → `[B,50,768]` |
| 06 | `__call__` | 取 CLS → 后置 Norm → `@ visual.proj` → 单位化 | `[B,50,768]` → `[B,512]` |

三个必须记住的模型约定：
1. 线性层权重存成 `[out,in]`，用 `W.T`；**唯一例外** `visual.proj` 已是 `[in,out]`，不转置。
2. 真实 `k` 是 4D，只能交换最后两个轴 `transpose(0,1,3,2)`，不能整体 `.T`。
3. 图块投影与最终投影**都没有偏置**；GELU 必须用课程提供的精确 erf 版本。

代码里每个 TODO 都写了逐步形状注释（中英混排），可直接对照 `PPT-CODE-MAP.txt` 的幻灯片编号。

## 4. 文件说明

| 文件 | 说明 |
|---|---|
| `image_encoder.py` | **本人完成的实现**（唯一修改的源文件） |
| `README.md` | 本文件（复现说明） |
| `README-course.md` | 课程原版 README 备份（未改动） |
| `report/实验报告.pdf` | **实验报告成品 PDF**（封面 + 3 页正文，实测共 4 页，符合 <5 页要求） |
| `report/实验报告.html` | 实验报告网页版（浏览器打开后 Ctrl+P 可再存成 PDF） |
| `report/overleaf/` | **基于深大官方模板的 Overleaf 版**（`main.tex` + `figs/`；模板来源见该目录 README） |
| `report/report.tex` | 自建 LaTeX 版（XeLaTeX + ctex + TikZ），作为 Overleaf 模板版的备选 |
| `report/实验报告.md` | 报告纯文本版，便于复制修改 |
| `report/figures/architecture.svg` | 架构插图（HTML / Markdown 引用） |
| `report/check_results.txt` | 数值检查原始输出 |
| `helpers.py`, `check.py`, `run.py`, `runtime.py`, `preprocess_image.py`, `transformer_steps.py` | 课程提供，未修改 |
| `weights/`, `assets/` | 课程提供的权重与参考数据，**不提交到 GitHub** |

## 5. 提交到 GitHub

```bash
git init
git add .gitignore README.md README-course.md image_encoder.py \
        helpers.py check.py run.py runtime.py preprocess_image.py \
        transformer_steps.py requirements.txt config.json \
        PPT-CODE-MAP.txt SOURCE-MANIFEST.json report/
git commit -m "Project 1: NumPy implementation of OpenCLIP ViT-B/32 image encoder"

# 方式 A：已登录 GitHub CLI 时，一条命令建仓并推送
gh repo create Project1-NumPy-ImageEncoder --public --source=. --remote=origin --push

# 方式 B：先在网页建空仓库，再手动关联
git remote add origin <你的仓库地址>
git branch -M main
git push -u origin main
```

推送成功后，把仓库链接填进报告封面与 Overleaf 版 `report/overleaf/main.tex` 的
`GitHub: <在此填入仓库链接>` 处。

`.gitignore` 已排除大文件：`weights/image_encoder.npz`、`assets/reference_outputs.npz`、
示例图片、`output_tokens.npz`、`__pycache__`。
**注意**：仓库里不含权重文件，别人复现时需要从课程附件自行放入 `weights/`
和 `assets/`（README 第 1 节已写明）。把仓库链接填进报告首页的 "GitHub" 一行。

## 6. 报告与声明

报告按课程要求控制在 **5 页以内（不含封面）**，包含：架构插图、红色高亮的
个人贡献、各阶段数值误差、一次形状变换的解释、一个真实修复的 bug、以及
贡献 / 协助来源声明。**AI 辅助已如实写入报告第 5 节**：本实现在 AI 助手
（WorkBuddy / 腾讯混元）协助下完成，包括思路讲解、代码骨架与报告排版；
关键公式理解、逐阶段验证与结论由本人核对。

## 7. 复用到 final project

按课程 README 说明，把完成的 `image_encoder.py` 与课程提供的 `helpers.py`
复制到 final project 的 `clip.py` / `text_encoder.py` 同目录即可；
`ImageEncoder(weights)`、`LayerNorm`、`TransformerBlock(weights,prefix,heads)`
三个接口保持不变，`causal=` 开关为后续文本塔预留。
