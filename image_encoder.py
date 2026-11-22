"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.

整体思路（中文导读）
--------------------
这个文件把一张 [3,224,224] 的图片变成一行 512 维的"语义向量"。
全程只有 NumPy 数组运算，没有训练、没有梯度、没有 PyTorch。
权重文件 weights/image_encoder.npz 里存的是已经学好的 152 个常数矩阵，
我们要做的只是照着 Transformer 的公式，把矩阵乘法一步步算对。

数据流（形状随每一步变化）：
    [B,3,224,224]  输入图片（B=一次几张）
 -> [B,49,3072]    切成 7x7 个 32x32 图块，每块摊平
 -> [B,50,768]     图块投影 + 前置 CLS 行 + 位置编码
 -> [B,50,768]     12 个 TransformerBlock（形状不变）
 -> [B,768]        只取第 0 行（CLS）
 -> [B,512]        投影 + 单位长度归一化 = 最终输出

两个容易踩坑的约定（来自 README / config.json）：
  1. 线性层权重存成 [out, in]，所以要用 W.T（转置）再相乘。
  2. 最后的 visual.proj 例外，它已经是 [in, out] = [768,512]，不要转置。
所有中间数组都保持 float32，否则 check.py 的 dtype 检查会失败。
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    """多头注意力：把 768 维特征拆成 12 个"小组"，每组各算一次注意力。"""

    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D] = [2304,768]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]   = [2304]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]  = [768,768]
        self.out_bias = weights[prefix + '.out_proj.bias']      # [D]    = [768]
        self.heads = heads                                      # 12
        self.width = self.out_bias.shape[0]                      # D = 768
        self.head_dim = self.width // heads                      # 64

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        通俗解释：x 的每一行有 768 个数。把它切成 12 段，每段 64 个数，
        每个"头"只负责其中一段。reshape 先把 768 拆成 (12,64)，
        transpose 再把"头"这个轴挪到 token 轴前面，方便后面批量矩阵乘法。

        为什么是 (B,N,H,Dh) -> transpose(0,2,1,3) 而不是别的？
        因为 PyTorch nn.MultiheadAttention 的权重按
        [q 的 12 个头 | k 的 12 个头 | v 的 12 个头] 顺序拼接，
        即"头是外层循环"：第 0 个 64 维属于头 0，接着头 1……
        所以 reshape(B,N,12,64) 正好对应权重存储顺序，不能先按头分块再排。

        形状变化：[B,N,768] -> reshape -> [B,N,12,64] -> transpose -> [B,12,N,64]
        """
        # ---- TODO-02 ----
        B, N, D = x.shape
        x = x.reshape(B, N, self.heads, self.head_dim)  # [B,N,H,Dh]
        return x.transpose(0, 2, 1, 3)                  # [B,H,N,Dh]
        # -----------------

    def __call__(self, x, causal=False):
        """TODO-03. 一次完整的多头注意力计算（含输出投影）。

        对应课堂算例 transformer_steps.py 的 walk_block()，只是把
        2D 玩具矩阵换成 4D 真实数组，并补上"分头 / 合头"两步。
        """
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        # qkv = x @ W.T + b，一次算出 Q、K、V 三份（拼在最后一个轴上）。
        # [B,N,768] @ [768,2304] -> [B,N,2304]
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])
        # 此时 q,k,v 均为 [B,12,50,64]

        # ---- TODO-03a: 打分 scores = Q · K^T / sqrt(head_dim) ----
        # k.transpose(0,1,3,2) 只交换最后两个轴（4D 版"K.T"），
        # 保留 batch 与 head 两个分组轴不动。这是 README 特别提醒的点：
        # 课堂算例里 k 是 2D 所以直接 .T，真实 k 是 4D，不能整体转置。
        # 除以 sqrt(64)=8 是为了防止点积数值过大、softmax 饱和。
        # [B,12,50,64] @ [B,12,64,50] -> [B,12,50,50]
        scale = np.sqrt(np.float32(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        # ---------------------------------------------------------

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper
        # coefficients: [B,12,50,50]，每一行的 50 个系数和为 1（softmax）

        # ---- TODO-03b: 用系数加权求和 V，再把 12 个头拼回去 ----
        # context = softmax(QK^T/sqrt(d)) @ V
        # [B,12,50,50] @ [B,12,50,64] -> [B,12,50,64]
        context = coefficients @ v

        # 合头 = split_heads 的逆操作：先把 H 轴换回 N 后面，再展平。
        # transpose(0,2,1,3): [B,12,50,64] -> [B,50,12,64]
        # reshape:            [B,50,12,64] -> [B,50,768]
        B, H, N, Dh = context.shape
        merged = context.transpose(0, 2, 1, 3).reshape(B, N, H * Dh)
        # --------------------------------------------------------

        # ---- TODO-03c: 输出投影（把 12 个头的结果再混合一次）----
        # 权重是 [out,in]，所以用 .T；偏置按行广播。
        # [B,50,768] @ [768,768] -> [B,50,768]
        return merged @ self.out_weight.T + self.out_bias
        # --------------------------------------------------------


class TransformerBlock:
    """一个 Transformer 块 = 注意力 + 两次残差 + MLP。形状始终 [B,N,768]。"""

    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D] = [3072,768]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']      # [3072]
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D] = [768,3072]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']      # [768]
        self.gelu = GELU()  # 老师提供的精确 GELU（erf 版），必须用它

    def __call__(self, x, causal=False):
        """前向：x -> Norm1 -> Attention -> +残差 -> Norm2 -> MLP -> +残差。

        "残差"就是"把原始输入直接加回来"：x + update。
        它的意义是让信息有一条不被矩阵乘法改写的捷径，
        深层网络（这里 12 层）才不会因为反复变换而丢失原始信号。
        注意：Norm 读的是 x，但相加要用**未归一化的原始 x**。
        """
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)  # [B,N,768]

        # ---- TODO-04a: 第一次残差相加 ----
        residual = x + update
        # ----------------------------------

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)  # 变量名 u 与课堂算例 / 参考实现保持一致

        # ---- TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2 ----
        # MLP 是"先放大 4 倍再缩回来"的两层全连接：
        #   hidden:     [B,50,768]  @ [768,3072] -> [B,50,3072]
        #   activated:  逐元素 GELU（用 helpers.GELU，别用近似 tanh 版）
        #   mlp_update: [B,50,3072] @ [3072,768] -> [B,50,768]
        # 最后再加第二次残差（加回 residual，不是加回 u，也不是加回 x）。
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update
        # ----------------------------------------------------------


class ImageEncoder:
    """完整的图像塔：切块 -> 嵌入 -> 12 层 -> CLS -> 512 维输出。"""

    def __init__(self, weights):
        # conv1 权重原本是 [768,3,32,32]，展平成 [768,3072] 后就是一个
        # 没有偏置的全连接层：每个 32x32 图块 -> 768 维 token。
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        通俗解释：把 224x224 的图按 32 像素切成 7x7=49 个小方块。
        每个小方块是 3 通道 x 32 高 x 32 宽 = 3072 个数，摊成一行。

        形状推导（这是报告里要讲的那次"形状变换"）：
          [B,3,224,224]
          -> reshape [B,3, 7,32, 7,32]   把 224 拆成 (7 行块, 块内 32 行)
          -> transpose(0,2,4,1,3,5)
             = [B,7,7,3,32,32]          把"块的行号/列号"挪到前面
          -> reshape [B,49,3072]        块按 (行,列) 排成一列

        transpose 的轴顺序 (0,2,4,1,3,5) 保证：
          - 先按"第几行块"再按"第几列块"遍历 => 从左到右、再从上到下；
          - 块内部按 C,H,W 摊平 => 与 conv1 权重 [768,3,32,32] 的展平顺序一致。
        """
        # ---- TODO-01 ----
        B, C, H, W = images.shape
        p = 32
        gh, gw = H // p, W // p          # 7, 7
        x = images.reshape(B, C, gh, p, gw, p)
        x = x.transpose(0, 2, 4, 1, 3, 5)   # [B,gh,gw,C,p,p]
        return x.reshape(B, gh * gw, C * p * p)  # [B,49,3072]
        # -----------------

    def embed(self, images):
        """TODO-05. 图块 -> token，并拼上 CLS 与位置编码 -> [B,50,768]。"""
        patches = self.patchify(images)  # [B,49,3072]

        # ---- TODO-05a: 图块投影，无偏置 ----
        # [B,49,3072] @ [3072,768] -> [B,49,768]
        # 注意 patch_weight 已经是 [768,3072]（[out,in]），所以用 .T。
        patch_tokens = patches @ self.patch_weight.T
        # ------------------------------------

        # Given: one supplied CLS row per image.
        # CLS 是一个可学习的"空白摘要行"，不对应任何图块，
        # 经过 12 层注意力后它会汇总整张图的信息，最后只取它输出。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))  # [B,1,768]

        # ---- TODO-05b: 前置 CLS + 加位置编码 ----
        # concatenate 在 token 轴（axis=1）上拼：1 + 49 = 50 行。
        # position 形状 [50,768]，靠 NumPy 广播自动加到每个 batch 上，
        # 作用是告诉模型"每个 token 原本在图片里的哪个位置"。
        tokens = np.concatenate([cls, patch_tokens], axis=1)  # [B,50,768]
        return tokens + self.position
        # -----------------------------------------

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)  # [B,50,768]

        # ---- TODO-06: 取 CLS -> 后置 Norm -> 投影 -> 单位化 ----
        # 1) 只取第 0 行 CLS：[B,50,768] -> [B,768]
        cls_token = tokens[:, 0, :]
        # 2) 后置 LayerNorm（和前面的 Norm 是**不同**的一组参数）
        #    LayerNorm 沿最后一轴求均值/方差，对 [B,768] 同样适用。
        normalized = self.post_norm(cls_token)
        # 3) 投影到 512 维。visual.proj 已经是 [in,out]=[768,512]，
        #    千万不要转置（这是与前面所有线性层唯一不同的地方，也无偏置）。
        #    [B,768] @ [768,512] -> [B,512]
        output = normalized @ self.projection
        # --------------------------------------------------------

        # 4) 老师提供的收尾：把每行缩放成单位长度（L2 归一化），
        #    这样向量之间做点积就是余弦相似度，便于图文匹配。
        return unit_vector(output)
