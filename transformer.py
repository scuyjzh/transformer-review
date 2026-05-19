import collections
import math
import time
import numpy as np
import torch
from torch import nn
from torch.utils import data


class DotProductAttention(nn.Module):
    """缩放点积注意力 Scaled Dot-Product Attention。

    该模块根据 queries 和 keys 计算注意力分数，
    再经过 masked_softmax 得到注意力权重，
    最后用注意力权重对 values 加权求和。

    注意力公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    其中:
        d 是 query/key 的特征维度。
    """

    def __init__(self, dropout, **kwargs):
        """初始化缩放点积注意力模块。

        参数:
            dropout: Dropout 概率，用于对注意力权重进行随机丢弃。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """执行缩放点积注意力前向传播。

        参数:
            queries: 查询张量。
                shape 为 (batch_size, num_queries, d)。

            keys: 键张量。
                shape 为 (batch_size, num_key_value_pairs, d)。

            values: 值张量。
                shape 为 (batch_size, num_key_value_pairs, value_dim)。

            valid_lens: 有效长度。
                用于在 softmax 前屏蔽无效位置，例如 padding。
                shape 可以为:
                    (batch_size,)
                    或 (batch_size, num_queries)。

        返回:
            注意力输出。
            shape 为 (batch_size, num_queries, value_dim)。
        """

        # d 是 query/key 的特征维度
        d = queries.shape[-1]

        # 计算注意力分数:
        # queries shape: (batch_size, num_queries, d)
        # keys.transpose(1, 2) shape: (batch_size, d, num_key_value_pairs)
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)

        # 对注意力分数做 masked softmax，得到注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 用注意力权重对 values 加权求和
        # output shape: (batch_size, num_queries, value_dim)
        return torch.bmm(self.dropout(self.attention_weights), values)

def masked_softmax(X, valid_lens):
    """带 mask 的 softmax。

    在最后一个维度上做 softmax，同时根据 valid_lens 屏蔽无效位置。
    被屏蔽的位置会先被赋值为一个很大的负数，使其 softmax 结果接近 0。

    参数:
        X: 输入张量。
            通常是注意力分数。
            shape 一般为 (batch_size, num_queries, num_key_value_pairs)。

        valid_lens: 有效长度。
            如果为 None，则直接对 X 的最后一维做 softmax。
            如果是一维张量，shape 为 (batch_size,)。
            如果是二维张量，shape 为 (batch_size, num_queries)。

    返回:
        masked softmax 后的张量。
        shape 与 X 相同。
    """

    # 如果没有提供有效长度，则直接对最后一维做 softmax
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)

    else:
        shape = X.shape

        # 如果 valid_lens 是一维:
        # valid_lens 的形状是 (batch_size,)
        # 需要为每个 query 复制一份有效长度
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        
        # 如果 valid_lens 是二维:
        # valid_lens 的形状是 (batch_size, num_queries)
        # 拉平成一维，方便和 X.reshape(-1, shape[-1]) 对应
        else:
            valid_lens = valid_lens.reshape(-1)
        
        # 将 X 变成二维:
        # X.reshape(-1, shape[-1]) 的形状是 (batch_size * num_queries, num_key_value_pairs)
        #
        # 然后根据 valid_lens 屏蔽每一行中超过有效长度的位置
        X = sequence_mask(
            X.reshape(-1, shape[-1]),
            valid_lens,
            value=-1e6
        )

        # 恢复原始 shape，并在最后一维做 softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def sequence_mask(X, valid_len, value=0):
    """根据有效长度屏蔽序列中的无效位置。

    参数:
        X: 输入张量。
            shape 通常为 (batch_size, num_steps)。

        valid_len: 每条序列的有效长度。
            shape 为 (batch_size,)。

        value: 被 mask 位置要填充的值，默认是 0。
            在 masked_softmax 中通常使用 -1e6。

    返回:
        被 mask 后的 X。
        超过 valid_len 的位置会被替换为 value。
    """

    # 序列最大长度
    maxlen = X.size(1)
    
    # 构造 mask
    # torch.arange(maxlen)[None, :] 形状是 (1, maxlen)
    # valid_len[:, None] 形状是 (batch_size, 1)
    # 广播比较后 mask 形状是 (batch_size, maxlen)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    
    # 将无效位置赋值为 value
    X[~mask] = value

    return X

class MultiHeadAttention(nn.Module):
    """多头注意力 Multi-Head Attention。

    多头注意力会先将 queries、keys、values 投影到 num_hiddens 维，
    然后拆分成 num_heads 个头并行计算注意力，
    最后再将多个头的输出拼接并通过 W_o 融合。

    输入输出核心 shape:
        输入 queries: (batch_size, num_queries, query_size)
        输入 keys:    (batch_size, num_key_value_pairs, key_size)
        输入 values:  (batch_size, num_key_value_pairs, value_size)

        输出:         (batch_size, num_queries, num_hiddens)
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        """初始化多头注意力模块。

        参数:
            key_size: key 输入维度。
            query_size: query 输入维度。
            value_size: value 输入维度。
            num_hiddens: 注意力输出总维度。
                也就是所有 head 拼接后的维度。
            num_heads: 注意力头数。
                要求 num_hiddens 能够被 num_heads 整除。
            dropout: Dropout 概率。
            bias: 线性层是否使用 bias，默认 False。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        # 把 queries，keys，values 分别线性投影到 num_hiddens 维
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

        # 多头拼接后再做一次线性变换，目的是重新融合多头信息，输出的维度还是 num_hiddens
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """执行多头注意力前向传播。

        参数:
            queries: 查询张量。
                shape 为 (batch_size, num_queries, query_size)。

            keys: 键张量。
                shape 为 (batch_size, num_key_value_pairs, key_size)。

            values: 值张量。
                shape 为 (batch_size, num_key_value_pairs, value_size)。

            valid_lens: 有效长度。
                用于屏蔽 padding 或未来 token。
                shape 可以为:
                    (batch_size,)
                    或 (batch_size, num_queries)。

        返回:
            多头注意力输出。
            shape 为 (batch_size, num_queries, num_hiddens)。
        """

        # 线性投影后再拆成多个头
        # 输出形状是：(batch_size*num_heads, num_queries, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # 因为 batch 维被扩展成 batch_size * num_heads，
        # 所以 valid_lens 也需要复制 num_heads 份
        if valid_lens is not None:
            # 复制后 valid_lens 的形状是 (batch_size*num_heads,) 或 (batch_size*num_heads, num_queries)
            valid_lens = torch.repeat_interleave(
                valid_lens,
                repeats=self.num_heads,
                dim=0
            )

        # 对每个 head 并行计算缩放点积注意力
        # output 的形状是 (batch_size*num_heads, num_queries, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 将多个 head 的输出重新拼接回 num_hiddens 维
        # output_concat 的形状是 (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        
        # 最后经过 W_o 线性变换后返回的形状是 (batch_size, num_queries, num_hiddens)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """为多头注意力的并行计算变换张量形状。

    该函数会把 num_hiddens 维拆分成:
        num_heads 个 head，每个 head 的维度为 num_hiddens / num_heads。

    参数:
        X: 输入张量。
            shape 为 (batch_size, num_steps, num_hiddens)。

        num_heads: 注意力头数。

    返回:
        变换后的张量。
        shape 为:
            (batch_size * num_heads, num_steps, num_hiddens / num_heads)。
    """

    # 将最后一维 num_hiddens 拆成:
    # num_heads 和 num_hiddens / num_heads
    #
    # 输入 X 的形状是 (batch_size, num_steps, num_hiddens)
    # 输出 X 的形状是 (batch_size, num_steps, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 调整维度顺序，使每个 head 都能拿到完整序列
    #
    # 输出 X 的形状是 (batch_size, num_heads, num_steps, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 将 num_heads 合并到 batch 维，方便并行计算注意力
    #
    # 最终输出 X 的形状是 (batch_size*num_heads, num_steps, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """还原 transpose_qkv 的形状变换。

    该函数会把多个 head 的输出重新拼接回 num_hiddens 维。

    参数:
        X: 多头注意力计算后的输出。
            shape 为:
                (batch_size * num_heads, num_queries, num_hiddens / num_heads)。

        num_heads: 注意力头数。

    返回:
        拼接后的输出张量。
        shape 为:
            (batch_size, num_queries, num_hiddens)。
    """

    # 先把 batch_size 和 num_heads 拆开
    # 输入 X 的形状是 (batch_size*num_heads, num_queries, num_hiddens/num_heads)
    #
    # 输出 X 的形状是 (batch_size，num_heads, num_queries, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    
    # 调整维度顺序，把 num_queries 放回第二维
    #
    # 输出 X 的形状是 (batch_size, num_queries, num_heads，num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    
    # 将多个 head 的最后一维拼接起来
    #
    # 最终输出 X 的形状是 (batch_size, num_queries, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络(Position-wise Feed-Forward Network)。

    Transformer 中的前馈网络会对序列中每个位置的 token 表示独立进行相同的非线性变换。

    结构:
        Linear(ffn_num_input -> ffn_num_hiddens)
        ReLU
        Linear(ffn_num_hiddens -> ffn_num_outputs)

    输入:
        X shape: (batch_size, num_steps, ffn_num_input)

    输出:
        output shape: (batch_size, num_steps, ffn_num_outputs)
    """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        """初始化基于位置的前馈网络。

        参数:
            ffn_num_input: 前馈网络输入维度。
            ffn_num_hiddens: 前馈网络隐藏层维度。
            ffn_num_outputs: 前馈网络输出维度。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(PositionWiseFFN, self).__init__(**kwargs)

        # 第一层线性变换: ffn_num_input -> ffn_num_hiddens
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)

        # 非线性激活函数
        self.relu = nn.ReLU()

        # 第二层线性变换: ffn_num_hiddens -> ffn_num_outputs
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """执行前馈网络前向传播。

        参数:
            X: 输入张量。
                shape 为 (batch_size, num_steps, ffn_num_input)。

        返回:
            输出张量。
                shape 为 (batch_size, num_steps, ffn_num_outputs)。
        """

        # 对每个位置的 token 表示独立进行相同的前馈网络变换
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层归一化。

    AddNorm 是 Transformer 中常见结构:

        output = LayerNorm(X + Dropout(Y))

    其中:
        X 是子层输入;
        Y 是子层输出, 例如注意力层输出或 FFN 输出。

    要求:
        X 和 Y 的 shape 必须相同, 才能做残差连接。
    """

    def __init__(self, normalized_shape, dropout, **kwargs):
        """初始化 AddNorm 模块。

        参数:
            normalized_shape: LayerNorm 的归一化维度。
                通常为 [num_hiddens] 或 num_hiddens。
            dropout: Dropout 概率。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(AddNorm, self).__init__(**kwargs)

        # 对子层输出 Y 做 dropout
        self.dropout = nn.Dropout(dropout)

        # 对残差连接后的结果做 LayerNorm
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """执行残差连接和层归一化。

        参数:
            X: 子层输入张量。
                shape 通常为 (batch_size, num_steps, num_hiddens)。
            Y: 子层输出张量。
                shape 必须与 X 相同。

        返回:
            经过 Dropout、残差连接和 LayerNorm 后的张量。
            shape 与 X 相同。
        """

        # 先对 Y 做 dropout, 再与 X 残差相加, 最后做 LayerNorm
        return self.ln(self.dropout(Y) + X)

class PositionalEncoding(nn.Module):
    """位置编码。

    由于 Transformer 本身没有 RNN 或 CNN 的顺序结构,
    所以需要给 token embedding 加入位置信息。

    这里使用固定的正弦/余弦位置编码:
        偶数维使用 sin;
        奇数维使用 cos。

    输入:
        X shape: (batch_size, num_steps, num_hiddens)

    输出:
        output shape: (batch_size, num_steps, num_hiddens)
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """初始化位置编码。

        参数:
            num_hiddens: 隐藏维度, 也就是 token embedding 的维度。
            dropout: Dropout 概率。
            max_len: 预先创建的位置编码最大长度。
                输入序列长度不能超过 max_len。
        """
        super(PositionalEncoding, self).__init__()

        # 对加了位置编码后的输入做 dropout
        self.dropout = nn.Dropout(dropout)

        # 创建足够长的位置编码矩阵 P
        # P shape: (1, max_len, num_hiddens)
        # 第 0 维为 1, 是为了后续和 batch 维度广播相加
        self.P = torch.zeros((1, max_len, num_hiddens))

        # 构造不同位置、不同频率对应的角度值
        # torch.arange(max_len).reshape(-1, 1) shape: (max_len, 1)
        # torch.arange(0, num_hiddens, 2) shape: (num_hiddens / 2,)
        # 通过广播相除后, X shape: (max_len, num_hiddens / 2)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        # 给偶数维填 sin
        self.P[:, :, 0::2] = torch.sin(X)

        # 给奇数维填 cos
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """给输入 token embedding 添加位置编码。

        参数:
            X: 输入张量。
                shape 为 (batch_size, num_steps, num_hiddens)。

        返回:
            加入位置编码并经过 dropout 后的张量。
                shape 仍为 (batch_size, num_steps, num_hiddens)。
        """

        # 根据当前输入序列长度 X.shape[1], 取出对应长度的位置编码
        # self.P[:, :X.shape[1], :] shape: (1, num_steps, num_hiddens)
        # 与 X 通过广播机制相加
        X = X + self.P[:, :X.shape[1], :].to(X.device)

        # 返回加了位置编码后的结果
        return self.dropout(X)

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基础封装类。

    该类将 encoder 和 decoder 组合在一起:
        1. encoder 负责对源序列进行编码;
        2. decoder 根据 encoder 的输出和解码器输入生成目标序列预测结果。
    """

    def __init__(self, encoder, decoder, **kwargs):
        """初始化 EncoderDecoder 模型。

        参数:
            encoder: 编码器模块, 例如 TransformerEncoder。
            decoder: 解码器模块, 例如 TransformerDecoder。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """执行编码器-解码器模型的前向传播。

        参数:
            enc_X: 编码器输入, 通常是源语言 token id 张量。
                shape 通常为 (batch_size, src_num_steps)。
            dec_X: 解码器输入, 通常是目标语言右移后的 token id 张量。
                shape 通常为 (batch_size, tgt_num_steps)。
            *args: 额外参数, 通常包括源序列有效长度 enc_valid_lens。

        返回:
            解码器输出和解码器状态。
            对 Transformer 来说, 输出 shape 通常为:
                (batch_size, tgt_num_steps, tgt_vocab_size)
        """

        # 编码器对源语言序列进行编码
        enc_outputs = self.encoder(enc_X, *args)

        # 根据编码器输出初始化解码器状态
        dec_state = self.decoder.init_state(enc_outputs, *args)

        # 解码器根据 dec_X 和 dec_state 生成预测结果
        return self.decoder(dec_X, dec_state)

class Encoder(nn.Module):
    """编码器基础接口。

    该类只定义编码器应该具备的 forward 方法,
    具体编码逻辑需要由子类实现。
    """

    def __init__(self, **kwargs):
        """初始化基础编码器。"""
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        """执行编码器前向传播。

        参数:
            X: 编码器输入。
            *args: 其他额外参数, 例如有效长度 valid_lens。

        返回:
            编码器输出。

        说明:
            该方法需要由具体子类实现。
        """
        raise NotImplementedError

class EncoderBlock(nn.Module):
    """Transformer 编码器块。

    一个 EncoderBlock 包含:
        1. 多头自注意力层;
        2. 第一次 AddNorm, 即 Dropout + 残差连接 + LayerNorm;
        3. 基于位置的前馈网络 PositionWiseFFN;
        4. 第二次 AddNorm。

    输入和输出 shape 保持一致:
        (batch_size, num_steps, num_hiddens)
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        """初始化 Transformer 编码器块。

        参数:
            key_size: key 输入维度。
            query_size: query 输入维度。
            value_size: value 输入维度。
            num_hiddens: Transformer 隐藏维度, 也是注意力输出维度。
            norm_shape: LayerNorm 的归一化维度, 通常为 [num_hiddens]。
            ffn_num_input: 前馈网络输入维度。
            ffn_num_hiddens: 前馈网络隐藏层维度。
            num_heads: 多头注意力的头数。
            dropout: Dropout 概率。
            use_bias: 线性层是否使用 bias。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(EncoderBlock, self).__init__(**kwargs)

        # 解码器自注意力层 self-attention
        # queries, keys, values 都来自同一个 X
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias
        )

        # 自注意力层后进行第一次残差连接和层规一化
        self.addnorm1 = AddNorm(norm_shape, dropout)

        # 基于位置的前馈网络
        # 对每个 token 的隐藏向量独立做非线性变换
        # 使用 ReLU 激活函数，输出维度和输入维度相同
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )

        # 前馈网络层后进行第二次残差连接和层规一化
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """执行 EncoderBlock 前向传播。

        参数:
            X: 编码器块输入。
                shape 为 (batch_size, num_steps, num_hiddens)。
            valid_lens: 源序列有效长度。
                用于在自注意力中屏蔽 padding 位置。
                shape 通常为 (batch_size,) 或 (batch_size, num_queries)。

        返回:
            当前 EncoderBlock 的输出。
            shape 为 (batch_size, num_steps, num_hiddens)。
        """

        # 多头自注意力:
        # self.attention(X, X, X, valid_lens)
        # 表示 queries, keys, values 都来自编码器输入 X
        # 输出 shape 仍然为 (batch_size, num_steps, num_hiddens)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))

        # 前馈网络 + 残差连接 + LayerNorm
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(Encoder):
    """Transformer 编码器。

    该编码器由以下部分组成:
        1. token embedding 词嵌入层;
        2. positional encoding 位置编码层;
        3. 多个 EncoderBlock 堆叠。

    输入:
        token id 序列, shape 为 (batch_size, num_steps)。

    输出:
        源序列的上下文表示, shape 为 (batch_size, num_steps, num_hiddens)。
    """

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        """初始化 Transformer 编码器。

        参数:
            vocab_size: 源语言词表大小。
            key_size: key 输入维度。
            query_size: query 输入维度。
            value_size: value 输入维度。
            num_hiddens: Transformer 隐藏维度, 也是词嵌入维度。
            norm_shape: LayerNorm 的归一化维度, 通常为 [num_hiddens]。
            ffn_num_input: 前馈网络输入维度。
            ffn_num_hiddens: 前馈网络隐藏层维度。
            num_heads: 多头注意力的头数。
            num_layers: EncoderBlock 的层数。
            dropout: Dropout 概率。
            use_bias: 多头注意力中的线性层是否使用 bias。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(TransformerEncoder, self).__init__(**kwargs)

        # 保存 num_hiddens，后面会用于 embedding 缩放
        self.num_hiddens = num_hiddens

        # 词嵌入层:
        # 输入 token id, 输出 token 向量
        # 输入 shape: (batch_size, num_steps)
        # 输出 shape: (batch_size, num_steps, num_hiddens)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # 位置编码层：给每个 token 向量加上位置信息
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        # 根据 num_layers 创建多个 EncoderBlock，并用 nn.Sequential 堆叠起来
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i),
                EncoderBlock(
                    key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, use_bias
                )
            )

    def forward(self, X, valid_lens, *args):
        """执行 Transformer 编码器前向传播。

        参数:
            X: 源语言 token id 序列。
                shape 为 (batch_size, num_steps)。
            valid_lens: 源语言序列有效长度。
                用于在注意力中屏蔽 padding 位置。
                shape 通常为 (batch_size,)。
            *args: 其他额外参数。

        返回:
            X: 编码器输出。
                shape 为 (batch_size, num_steps, num_hiddens)。
        """

        # token id -> embedding
        # 然后乘以 sqrt(num_hiddens), 再加位置编码
        # 乘 sqrt(num_hiddens) 是为了让 embedding 的数值尺度和位置编码更匹配
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 初始化 attention_weights，保存每一层 EncoderBlock 的注意力权重
        self.attention_weights = [None] * len(self.blks)
        
        # 依次通过每个 EncoderBlock
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            
            # 保存第 i 层 EncoderBlock 的多头自注意力权重
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X

class Decoder(nn.Module):
    """编码器-解码器架构中的基础解码器接口。

    该类只定义了解码器应该具备的基本方法,
    具体的解码逻辑需要由子类实现。
    """

    def __init__(self, **kwargs):
        """初始化基础解码器。"""
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """根据编码器输出初始化解码器状态。

        参数:
            enc_outputs: 编码器的输出。
            *args: 其他额外参数, 例如有效长度 enc_valid_lens。

        返回:
            解码器初始状态。

        说明:
            该方法需要由具体子类实现。
        """
        raise NotImplementedError

    def forward(self, X, state):
        """执行解码器前向传播。

        参数:
            X: 解码器输入。
            state: 解码器状态。

        返回:
            解码器输出和更新后的状态。

        说明:
            该方法需要由具体子类实现。
        """
        raise NotImplementedError

class AttentionDecoder(Decoder):
    """带注意力机制的解码器基础接口。

    在普通 Decoder 的基础上, 额外规定子类需要提供 attention_weights 属性,
    用于访问解码过程中的注意力权重。
    """

    def __init__(self, **kwargs):
        """初始化带注意力机制的解码器。"""
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        """返回解码器中的注意力权重。

        返回:
            attention_weights: 注意力权重。

        说明:
            该属性需要由具体子类实现。
        """
        raise NotImplementedError

class DecoderBlock(nn.Module):
    """Transformer 解码器中的单个 DecoderBlock。

    一个 DecoderBlock 包含三部分:
        1. Masked Self-Attention, 解码器自注意力;
        2. Cross-Attention, 编码器-解码器交叉注意力;
        3. PositionWiseFFN, 基于位置的前馈网络。

    每个子层后面都会接 AddNorm, 即 Dropout + 残差连接 + LayerNorm。
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        """初始化 Transformer 解码器块。

        参数:
            key_size: key 输入维度。
            query_size: query 输入维度。
            value_size: value 输入维度。
            num_hiddens: Transformer 隐藏维度, 也是注意力输出维度。
            norm_shape: LayerNorm 归一化维度, 通常为 [num_hiddens]。
            ffn_num_input: 前馈网络输入维度。
            ffn_num_hiddens: 前馈网络隐藏层维度。
            num_heads: 多头注意力的头数。
            dropout: Dropout 概率。
            i: 当前 DecoderBlock 的层编号, 用于索引 state[2][i]。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(DecoderBlock, self).__init__(**kwargs)

        # 表示当前是第几个解码器块，后面会用到它来索引 state[2]，以获得直到当前时间步第i个块解码的输出表示
        self.i = i

        # 第一层注意力：解码器自注意力 Masked self-attention
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)

        # 自注意力层后进行第一次残差连接和层规一化
        self.addnorm1 = AddNorm(norm_shape, dropout)

        # 第二层注意力：编码器－解码器交叉注意力 Cross-Attention
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)

        # 交叉注意力层后进行第二次残差连接和层规一化

        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 基于位置的前馈网络层，使用 ReLU 激活函数，输出维度和输入维度相同
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)

        # 前馈网络层后进行第三次残差连接和层规一化
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """执行单个 DecoderBlock 的前向传播。

        参数:
            X: 当前 DecoderBlock 的输入。
                训练时 shape 为 (batch_size, num_steps, num_hiddens)。
                预测时通常 shape 为 (batch_size, 1, num_hiddens)。

            state: 解码器状态列表。
                state[0]: enc_outputs, 编码器输出,
                    shape 为 (batch_size, src_num_steps, num_hiddens)。
                state[1]: enc_valid_lens, 编码器有效长度,
                    shape 通常为 (batch_size,)。
                state[2]: 每层 DecoderBlock 的历史缓存列表,
                    长度为 num_layers。
                    state[2][self.i] 保存当前层已经解码过的历史表示。

        返回:
            output: 当前 DecoderBlock 的输出,
                shape 为 (batch_size, num_steps, num_hiddens)。
            state: 更新后的解码器状态。
        """
        enc_outputs, enc_valid_lens = state[0], state[1]

        # 给 Decoder 的 自注意力层 attention1 准备 keys 和 values
        if state[2][self.i] is None:
            # 如果当前层还没有历史缓存：
            # 1. 训练阶段：通常一次性输入完整目标序列，此时缓存初始为 None
            # 2. 预测阶段第一个时间步：还没有历史 token，也为 None
            #
            # 因此直接把当前输入 X 作为 key_values
            # 训练时：X 是完整目标序列的表示，形状通常为 (batch_size, num_steps, hidden_dim)
            # 预测第一个时间步：X 是当前 token 的表示，形状通常为 (batch_size, 1, hidden_dim)
            key_values = X
        else:
            # 如果当前层已经有历史缓存：
            # 说明处于自回归预测阶段，并且已经生成过前面的 token
            #
            # state[2][self.i] 保存的是当前 DecoderBlock 之前时间步的输入表示
            # X 是当前时间步新输入 token 的表示
            #
            # 沿着序列长度维度 axis=1 进行拼接：
            # 历史 token 表示 + 当前 token 表示
            #
            # 拼接前：
            # state[2][self.i].shape = (batch_size, 历史长度, hidden_dim)
            # X.shape                = (batch_size, 1, hidden_dim)
            #
            # 拼接后：
            # key_values.shape       = (batch_size, 历史长度 + 1, hidden_dim)
            key_values = torch.cat((state[2][self.i], X), axis=1)

        # 更新当前 DecoderBlock 的历史缓存
        # 这样下一次预测新 token 时，就可以继续使用之前已经生成 token 的表示
        #
        # 注意：这里缓存的是当前 block 的输入表示，
        # 不是当前 block 完整计算后的输出。
        # 它会在当前层 self-attention 中作为 key 和 value 的来源。
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape

            # 训练时构造解码器自注意力 mask，
            # 第 t 个位置只能看见前 t 个位置, 不能看未来 token。
            # dec_valid_lens 的形状是 (batch_size, num_steps)，
            # 其中每一行都是[1,2,...,num_steps]，
            # 表示第1个 query 只能看前1个 key，第2个 query 只能看前2个 key，以此类推
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 预测时每次只输入当前 token, 没有未来 token, 因此不需要 mask
            dec_valid_lens = None

        # 1. 解码器 masked self-attention
        # X 的形状是 (batch_size, num_steps, num_hiddens)
        # key_values 的形状是 (batch_size, num_steps, num_hiddens)
        # 通过 dec_valid_lens，每个位置只能看到自己和前面的 token
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        
        # 2. 残差连接 + LayerNorm
        # X2 的形状是 (batch_size, num_steps, num_hiddens)
        Y = self.addnorm1(X, X2)

        # 3. 编码器-解码器交叉注意力
        # Y 的形状是 (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)

        # 4. 残差连接 + LayerNorm
        # Y2 的形状是 (batch_size, num_steps, num_hiddens)
        Z = self.addnorm2(Y, Y2)

        # 5. 前馈网络 + 残差连接 + LayerNorm
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(AttentionDecoder):
    """Transformer 解码器。

    该解码器由以下部分组成:
        1. 目标语言词嵌入层;
        2. 位置编码层;
        3. 多个 DecoderBlock;
        4. 输出层 dense, 将隐藏状态映射到目标词表大小。

    训练时:
        一次输入完整目标序列, 通过 mask 防止看到未来 token。

    预测时:
        从 <bos> 开始逐 token 生成, 并通过 state 缓存历史解码表示。
    """

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        """初始化 Transformer 解码器。

        参数:
            vocab_size: 目标语言词表大小。
            key_size: key 输入维度。
            query_size: query 输入维度。
            value_size: value 输入维度。
            num_hiddens: Transformer 隐藏维度, 也是词嵌入维度。
            norm_shape: LayerNorm 归一化维度, 通常为 [num_hiddens]。
            ffn_num_input: 前馈网络输入维度。
            ffn_num_hiddens: 前馈网络隐藏层维度。
            num_heads: 多头注意力头数。
            num_layers: DecoderBlock 层数。
            dropout: Dropout 概率。
            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(TransformerDecoder, self).__init__(**kwargs)

        # 保存 num_hiddens，后面会用于 embedding 缩放
        self.num_hiddens = num_hiddens
        
        self.num_layers = num_layers

        # 词嵌入层，把 token id 转成向量，形状是 (batch_size, num_steps, num_hiddens)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # 位置编码层，给每个 token 向量加上位置信息
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)

        # 根据 num_layers 创建多个 DecoderBlock，并用 nn.Sequential 堆叠起来
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block"+str(i),
                DecoderBlock(
                    key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, i
                )
            )

        # 输出层: num_hiddens -> vocab_size
        # 把每个目标位置的 num_hiddens 维隐藏表示转换成词表大小 vocab_size 的 logits
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """初始化 TransformerDecoder 的状态。

        参数:
            enc_outputs: 编码器输出,
                shape 为 (batch_size, src_num_steps, num_hiddens)。
            enc_valid_lens: 源语言序列有效长度,
                shape 通常为 (batch_size,)。
            *args: 其他额外参数。

        返回:
            state: 解码器状态列表。
                state[0]: enc_outputs, 编码器输出。
                state[1]: enc_valid_lens, 编码器有效长度。
                state[2]: 每一层 DecoderBlock 的历史缓存列表,
                    初始为 [None] * num_layers。
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """执行 TransformerDecoder 前向传播。

        参数:
            X: 解码器输入 token id。
                训练时 shape 为 (batch_size, tgt_num_steps)。
                预测时 shape 通常为 (batch_size, 1)。

            state: 解码器状态。
                由 init_state 返回, 包含编码器输出, 编码器有效长度,
                以及每个 DecoderBlock 的历史缓存。

        返回:
            output: 解码器输出 logits,
                shape 为 (batch_size, tgt_num_steps, vocab_size)。
            state: 更新后的解码器状态。
        """

        # token id -> embedding, 
        # 因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        # X 的形状是 (batch_size, tgt_num_steps, num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 初始化注意力权重保存列表，一共两行：
        # 第 0 行保存各层 masked self-attention 权重;
        # 第 1 行保存各层 cross-attention 权重。
        self._attention_weights = [
            [None] * len(self.blks) for _ in range (2)
        ]

        # 依次通过每个 DecoderBlock
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)

            # 保存第 i 层解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights

            # 保存第 i 层编码器-解码器交叉注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights

        # 将隐藏状态映射到目标词表大小
        return self.dense(X), state

    @property
    def attention_weights(self):
        """返回 TransformerDecoder 的注意力权重。

        返回:
            attention_weights: 二维列表。
                attention_weights[0][i]:
                    第 i 层 DecoderBlock 的 masked self-attention 权重。
                attention_weights[1][i]:
                    第 i 层 DecoderBlock 的 cross-attention 权重。
        """
        return self._attention_weights

class Vocab:
    """文本词表类。

    用于根据 token 序列构建词表, 并提供 token 和 index 之间的双向转换。

    主要功能:
        1. 统计 token 词频;
        2. 按词频从高到低排序;
        3. 构建 index -> token 的映射 idx_to_token;
        4. 构建 token -> index 的映射 token_to_idx;
        5. 支持通过 min_freq 过滤低频 token;
        6. 支持添加特殊 token, 例如 <pad>, <bos>, <eos>;
        7. 支持未知 token <unk>, 其索引固定为 0。
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表。

        参数:
            tokens: token 列表, 可以是一维列表或二维列表。
                一维示例:
                    ['i', 'love', 'you']
                二维示例:
                    [['i', 'love', 'you'], ['you', 'love', 'me']]

            min_freq: 最小词频阈值。
                只有出现次数大于等于 min_freq 的 token 才会被加入词表。

            reserved_tokens: 保留的特殊 token 列表。
                例如:
                    ['<pad>', '<bos>', '<eos>']

        生成的主要属性:
            self.idx_to_token:
                index -> token 的映射列表。
                例如:
                    ['<unk>', '<pad>', '<bos>', '<eos>', 'i', 'love']

            self.token_to_idx:
                token -> index 的映射字典。
                例如:
                    {'<unk>': 0, '<pad>': 1, 'i': 4}

            self._token_freqs:
                按词频从高到低排序后的 token 频率列表。
        """

        # 如果没有传入 tokens, 则使用空列表
        if tokens is None:
            tokens = []

        # 如果没有传入 reserved_tokens, 则使用空列表
        if reserved_tokens is None:
            reserved_tokens = []

        # 统计 token 词频
        counter = count_corpus(tokens)

        # 按词频从高到低排序
        self._token_freqs = sorted(
            counter.items(),     # counter.items() 中每个元素形如: (token, freq)，
            key=lambda x: x[1],  # key=lambda x: x[1] 表示按照元组的第二个元素（即词频）进行排序，
            reverse=True         # reverse=True 表示降序排序
        )

        # 创建编号 index → token 的映射。默认第 0 个 token 是 <unk>, 表示未知词
        # 假设 reserved_tokens = ['<pad>', '<bos>', '<eos>']，
        # 那么：self.idx_to_token = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.idx_to_token = ['<unk>'] + reserved_tokens

        # 创建 token → 编号 index 的反向映射, 
        # 例如：self.token_to_idx = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.idx_to_token)
        }

        # 遍历按词频排序后的 token, 将满足 min_freq 的 token 加入词表
        for token, freq in self._token_freqs:
            # 因为 _token_freqs 已经按词频降序排序,
            # 所以一旦当前 token 词频小于 min_freq, 后面的 token 也都会小于 min_freq
            if freq < min_freq:
                break

            # 避免重复添加 reserved_tokens 中已经存在的 token
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小。

        返回:
            词表中 token 的总数量。
        """
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """将 token 转换为对应的 index。

        参数:
            tokens: 单个 token 或 token 列表。
                例如:
                    'i'
                    ['i', 'love', 'you']

        返回:
            如果输入是单个 token, 返回对应的 index;
            如果输入是 token 列表, 返回 index 列表;
            如果 token 不在词表中, 返回 <unk> 的索引 0。
        """

        # 如果输入的是单个 token, 直接返回它的 index
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)

        # 如果输入的是 token 列表, 递归地把每个 token 转成 index
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """将 index 转换回对应的 token。

        参数:
            indices: 单个 index 或 index 列表。
                例如:
                    4
                    [4, 5, 6]

        返回:
            如果输入是单个 index, 返回对应的 token;
            如果输入是 index 列表, 返回 token 列表。
        """

        # 如果输入的是单个 index, 直接返回对应 token
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]

        # 如果输入的是 index 列表, 返回对应的 token 列表
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """返回未知 token <unk> 的索引。

        返回:
            <unk> 的索引, 固定为 0。
        """
        return 0

    @property
    def token_freqs(self):
        """返回 token 词频列表。

        返回:
            按词频从高到低排序后的列表。
            每个元素形如:
                (token, freq)
        """
        return self._token_freqs

def count_corpus(tokens):
    """统计 token 词频。

    参数:
        tokens: token 列表, 可以是一维列表或二维列表。
            一维示例:
                ['i', 'love', 'you', 'you']
            二维示例:
                [['i', 'love', 'you'], ['you', 'love', 'me']]

    返回:
        collections.Counter 对象。
        键是 token, 值是该 token 出现的次数。
        例如:
            Counter({'you': 2, 'i': 1, 'love': 1})
    """

    # 如果 tokens 是空列表, 或者 tokens 是二维列表,
    # 则先将二维 token 列表拉平成一维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 知识点：列表推导式的顺序规则
        # 列表推导式的多个 for，执行顺序和普通嵌套循环一致，从左到右看
        tokens = [token for line in tokens for token in line]

    # 统计每个 token 的出现次数，返回一个 dict，键是 token，值是频次
    return collections.Counter(tokens)

def try_gpu(i=0):
    """返回指定编号的 GPU 设备；如果不存在，则返回 CPU 设备。

    参数:
        i: GPU 设备编号，默认值为 0。
            例如 i=0 表示尝试使用 cuda:0；
            i=1 表示尝试使用 cuda:1。

    返回:
        torch.device 对象。
            如果指定编号的 GPU 存在，则返回 torch.device(f'cuda:{i}')；
            如果不存在，则返回 torch.device('cpu')。
    """

    # 如果当前机器上的 GPU 数量大于等于 i + 1，说明编号为 i 的 GPU 存在
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')

    # 如果指定 GPU 不存在，则使用 CPU
    return torch.device('cpu')

def preprocess_nmt():
    """读取并预处理英法翻译数据集。

    该函数会读取原始英法翻译文本, 并进行基础清洗:
        1. 将特殊空格替换为普通空格;
        2. 将所有英文字符转成小写;
        3. 在标点符号前添加空格, 方便后续按空格分词。

    返回:
        text: 预处理后的文本字符串。
            例如:
                "Go.\\tVa !"
            会被处理成:
                "go .\\tva !"
    """

    def read_data_nmt():
        """读取原始英法翻译文本文件。

        返回:
            文件中的全部文本内容。
        """
        with open('/root/d2l-zh/data/fra-eng/fra.txt', 'r') as f:
            return f.read()

    def no_space(char, prev_char):
        """判断当前标点符号前面是否缺少空格。

        参数:
            char: 当前字符。
            prev_char: 当前字符的前一个字符。

        返回:
            如果 char 是标点符号 , . ! ? 之一,
            并且 prev_char 不是空格, 则返回 True;
            否则返回 False。
        """
        return char in set(',.!?') and prev_char != ' '

    # 读取原始文本
    text = read_data_nmt()

    # 替换特殊空格'\u202f'和'\xa0', 并统一转成小写
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在标点前加空格, 使标点可以作为独立 token
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]

    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """将预处理后的机器翻译文本切分为源语言和目标语言的 token 序列。

    参数:
        text: 预处理后的机器翻译文本。
            每一行通常是一对句子，源语言和目标语言之间用制表符 '\t' 分隔。
            例如:
                "go .\\tva !\\n"
                "i love you .\\tje t'aime ."

        num_examples: 最多读取的样本数量。
            如果为 None，则读取全部样本；
            如果指定数值，则最多读取前 num_examples 条样本。

    返回:
        source: 源语言分词结果列表。
            每个元素是一条源语言句子的 token 列表。
            例如:
                [['go', '.'], ['i', 'love', 'you', '.']]

        target: 目标语言分词结果列表。
            每个元素是一条目标语言句子的 token 列表。
            例如:
                [['va', '!'], ['je', "t'aime", '.']]
    """

    source, target = [], []

    # 按换行符切分文本，每一行是一对翻译句子
    for i, line in enumerate(text.split('\n')):
        # 如果指定了 num_examples，则最多读取指定数量的样本
        if num_examples and i > num_examples:
            break

        # 每一行按制表符 '\t' 切分为源语言句子和目标语言句子
        parts = line.split('\t')

        # 只处理能够正确切分成 source 和 target 的句子对
        if len(parts) == 2:
            # 源语言句子按空格分词
            source.append(parts[0].split(' '))

            # 目标语言句子按空格分词
            target.append(parts[1].split(' '))
    return source, target

def build_array_nmt(lines, vocab, num_steps):
    """将分词后的文本序列转换为固定长度的 token id 张量，并计算有效长度。

    参数:
        lines: 分词后的文本序列列表。
            例如:
                [
                    ['i', 'love', 'you'],
                    ['go']
                ]
            其中每个元素是一条已经分好词的句子。

        vocab: 词表对象。
            用于将 token 转换成对应的 index。
            例如:
                vocab['i'] -> 4
                vocab['<pad>'] -> 1
                vocab['<eos>'] -> 3

        num_steps: 每条序列的固定长度。
            如果序列长度超过 num_steps，则截断；
            如果序列长度不足 num_steps，则用 <pad> 补齐。

    返回:
        array: 固定长度的 token id 张量。
            shape 为 (len(lines), num_steps)。

        valid_len: 每条序列的有效长度。
            shape 为 (len(lines),)。
            每个元素表示对应序列中非 <pad> token 的数量。
    """

    # 把每个句子的 token 列表转换成 index 数字编号列表
    lines = [vocab[l] for l in lines]

    # 给每个句子加 <eos>，表示 end of sequence，序列结束
    lines = [l + [vocab['<eos>']] for l in lines]

    # 对每条序列进行截断或补齐，使其长度统一为 num_steps
    # 然后转换成 tensor，最终得到的 array 的形状是 (len(lines), num_steps)
    array = torch.tensor([
        truncate_pad(l, num_steps, vocab['<pad>'])
        for l in lines])

    # 计算每条序列的有效长度
    # 非 <pad> 的位置为 True，<pad> 的位置为 False
    # 转成 int32 后，True -> 1，False -> 0
    # 最后沿着 dim=1 求和，得到每条序列中非 <pad> token 的数量
    # valid_len 的形状是 (len(lines),)，其中每个元素都是对应序列里，非 <pad> 的 token 数量
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)

    return array, valid_len

def truncate_pad(line, num_steps, padding_token):
    """将一条序列截断或补齐到固定长度。

    参数:
        line: 输入序列，通常是 token index 列表。
        num_steps: 目标序列长度。
            如果 line 长度大于 num_steps，则截断；
            如果 line 长度小于 num_steps，则用 padding_token 补齐。
        padding_token: 用于补齐序列的 token，通常是 <pad> 对应的索引。

    返回:
        长度为 num_steps 的序列。
    """

    # 如果序列长度超过 num_steps，则截断保留前 num_steps 个元素
    if len(line) > num_steps:
        return line[:num_steps]

    # 如果序列长度不足 num_steps，则在末尾补 padding_token，直到长度达到 num_steps
    return line + [padding_token] * (num_steps - len(line))

def load_array(data_arrays, batch_size, is_train=True):
    """将多个张量封装成 PyTorch 数据迭代器。

    参数:
        data_arrays: 由多个张量组成的元组或列表。
            这些张量的第 0 维长度必须一致，表示样本数量相同。
            例如: (src_array, src_valid_len, tgt_array, tgt_valid_len)。
        batch_size: 每个小批量 batch 中包含的样本数量。
        is_train: 是否用于训练。
            如果为 True，则 DataLoader 会打乱数据顺序；
            如果为 False，则不打乱数据顺序。

    返回:
        一个 PyTorch DataLoader 对象。
        每次迭代会返回一个 batch。
    """

    # TensorDataset 的作用是把多个张量按样本维度打包成一个数据集
    dataset = data.TensorDataset(*data_arrays)

    # DataLoader 会把 dataset 按小批量返回，并可以根据 is_train 决定是否打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """加载并构造机器翻译训练数据集。

    该函数会读取英法翻译原始文本, 完成文本预处理, 分词, 构建源语言和目标语言词表,
    并将文本序列转换为固定长度的 token index 张量, 最后封装成 PyTorch DataLoader。

    参数:
        batch_size: 每个 batch 中包含的样本数量。
        num_steps: 每条源语言和目标语言序列的固定长度。
            如果序列长度超过 num_steps, 则截断;
            如果序列长度不足 num_steps, 则用 <pad> 补齐。
        num_examples: 最多读取的样本数量, 默认值为 600。

    返回:
        data_iter: 训练数据迭代器。
            每次迭代返回四个张量:
            源语言序列, 源语言有效长度, 目标语言序列, 目标语言有效长度。
        src_vocab: 源语言词表。
        tgt_vocab: 目标语言词表。
    """

    # 读取并清洗文本：小写化、特殊空格替换、标点分离
    text = preprocess_nmt()

    # 将文本切分为源语言序列和目标语言序列
    source, target = tokenize_nmt(text, num_examples)

    # 构建源语言词表
    src_vocab = Vocab(
        source,
        min_freq=2,
        reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 构建目标语言词表
    tgt_vocab = Vocab(
        target,
        min_freq=2,
        reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 将源语言 token 转成 index, 添加 <eos>, 并 padding/truncate 到固定长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)

    # 将目标语言 token 转成 index, 添加 <eos>, 并 padding/truncate 到固定长度
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # 将四个张量打包成数据数组
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)

    # 构建 DataLoader 迭代器，返回四个张量：源语言序列、源语言有效长度、目标语言序列、目标语言有效长度
    data_iter = load_array(data_arrays, batch_size)

    return data_iter, src_vocab, tgt_vocab

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带 mask 的 softmax 交叉熵损失。

    用于序列预测任务，例如机器翻译、文本生成等。

    普通交叉熵会对所有时间步计算损失，包括 <pad> 位置；
    但 <pad> 只是为了补齐序列长度，不是真实标签，
    所以需要根据 valid_len 构造 mask，将 padding 位置的 loss 置为 0。

    参数:
        pred: 模型预测结果，shape 为 (batch_size, num_steps, vocab_size)。
            其中 vocab_size 表示目标词表大小，每个时间步都会输出一个词表分数向量。
        label: 真实标签，shape 为 (batch_size, num_steps)。
            每个元素是目标 token 的索引。
        valid_len: 每条序列的有效长度，shape 为 (batch_size,)。
            用于判断哪些位置是真实 token，哪些位置是 <pad>。

    返回:
        weighted_loss: 每条样本的 masked loss，shape 为 (batch_size,)。
            padding 位置的损失被置为 0。
    """
    def forward(self, pred, label, valid_len):
        # 掩码权重矩阵，标记有效位置，默认为 1。
        # weights 的形状是 (batch_size, num_steps)，与 label 的形状相同
        weights = torch.ones_like(label)
        # 根据 valid_len 屏蔽 padding 位置，padding 位置权重为 0
        weights = sequence_mask(weights, valid_len)
        # 普通交叉熵默认可能会直接求平均，返回一个标量。
        # 但这里不能直接平均，因为还要手动 mask padding 位置。
        # 所以设置：reduction = 'none'，表示每个样本、每个时间步的 loss 都保留下来
        self.reduction='none'
        # 计算未加权交叉熵
        # PyTorch 的 nn.CrossEntropyLoss 要求类别维度在第 1 维，
        # pred 原 shape: (batch_size, num_steps, vocab_size)
        # permute 后: (batch_size, vocab_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 乘 mask，将 padding 位置的损失置为 0，并对每条样本的时间步维度求平均
        # * 表示 逐元素相乘，mean(dim=1) 表示对时间步维度求平均
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # `weighted_loss` shape: (batch_size,)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练学列到序列模型。

    参数:
        net: seq2seq 模型，例如 EncoderDecoder(encoder, decoder)。
        data_iter: 训练数据迭代器，每次返回一个 batch。
        lr: 学习率，控制参数更新步长。
        num_epochs: 训练轮数。
        tgt_vocab: 目标语言词表，用来获取 <bos> 等特殊词元的编号。
        device: 训练设备，例如 cuda:0 或 cpu。
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            # Transformer 中需要 Xavier 初始化的线性层：
            # 1.多头注意力里的线性层(W_q/W_k/W_v/W_o)；
            # 2.前馈网络里的线性层(dense1/dense2)；
            # 3.解码器输出层的线性层(dense)。
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # 递归访问模型中的所有子模块。凡是遇到 nn.Linear，就用 Xavier 初始化它的权重参数
    net.apply(xavier_init_weights)
    # 把模型放到指定设备上，例如 GPU 或 CPU；每个 batch 也要移动到同一个设备上，才能进行计算
    net.to(device)
    # 定义 Adam 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 定义带 mask 的交叉熵损失函数，计算损失时会自动屏蔽掉 padding 位置
    loss = MaskedSoftmaxCELoss()
    # 让模型进入训练模式，启用 Dropout
    net.train()
    for epoch in range(num_epochs):
        # 统计训练速度
        timer = Timer()
        # metric[0]：loss 总和
        # metric[1]：有效 token 数量
        metric = Accumulator(2)
        for batch in data_iter:
            # 1.清空上一轮梯度
            optimizer.zero_grad()
            # 2.取出源句子和目标句子，把 batch 移动到设备上
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 构造 <bos> 的 batch，形状是 (batch_size, 1)，每个元素都是 <bos> 的编号
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            # 3.构造 Decoder 输入：把目标序列右移一位，在开头加 <bos>，作为 Decoder 的输入，来进行 Teacher Forcing
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 4.前向传播
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # 5.计算 masked loss
            l = loss(Y_hat, Y, Y_valid_len)
            # 6.反向传播。.sum()把 batch 内所有样本 loss 加起来，变成一个标量
            l.sum().backward()
            # 7.梯度裁剪。计算模型所有参数梯度的整体 L2 范数。如果梯度范数超过 1，就按比例缩小到 1
            grad_clipping(net, 1)
            # 统计有效 token 数，<pad> 不计入有效 token
            num_tokens = Y_valid_len.sum()
            # 8.参数更新
            optimizer.step()
            # 9.累计 loss 和 token 数。torch.no_grad() 表示这里不需要记录梯度，因为统计 loss 不参与训练
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            # 每 10 个 epoch 打印一次 loss
            print(f'epoch {epoch + 1:3d}, loss {metric[0] / metric[1]:.4f}')
    # 打印最后一个 epoch 的 loss 和训练速度
    print(f'loss {metric[0] / metric[1]:.4f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def grad_clipping(net, theta):
    """对模型参数进行梯度裁剪，防止梯度爆炸。

    参数:
        net: 神经网络模型。
            如果是 nn.Module 类型，则通过 net.parameters() 获取参数；
            否则默认模型中有 net.params 属性。
        theta: 梯度范数阈值。
            当所有参数梯度的整体 L2 范数大于 theta 时，
            按比例缩小所有梯度，使整体梯度范数不超过 theta。

    返回:
        None。该函数会直接原地修改参数的梯度 param.grad。
    """

    # 获取模型中所有需要训练的参数
    if isinstance(net, nn.Module):
        params = [
            p for p in net.parameters()
            if p.requires_grad and p.grad is not None
        ]
    else:
        params = [
            p for p in net.params
            if p.grad is not None
        ]

    # 计算所有参数梯度的 L2 范数
    # norm = sqrt(sum(g_i^2))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 如果梯度范数超过阈值 theta，则对所有梯度按同一比例缩放
    # 缩放后新的整体梯度范数约等于 theta
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """使用训练好的 seq2seq 模型进行预测翻译。

    参数:
        net: 已训练好的 seq2seq 模型，例如 EncoderDecoder(encoder, decoder)。
        src_sentence: 待翻译的源语言句子，字符串形式，例如 'go .'。
        src_vocab: 源语言词表，用来把源语言 token 转成 index。
        tgt_vocab: 目标语言词表，用来把预测出的 index 转回 token。
        num_steps: 最大序列长度；源句子会被截断/补齐到该长度，预测时最多生成该长度的 token。
        device: 计算设备，例如 cuda:0 或 cpu。
        save_attention_weights: 是否保存预测过程中的注意力权重，默认 False。

    返回:
        翻译后的目标语言句子，以及预测过程中保存的注意力权重列表。
    """
    # 把模型切换到预测模式，关闭 Dropout
    net.eval()
    # 源句子转小写、按空格分词，再通过源语言词表 src_vocab 转成 token index，最后加 <eos>
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    # 构造源句子的有效长度，enc_valid_len 的形状是 (1,)，1 是 batch_size，因为预测时通常一次只翻译一句
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 把源序列截断或补齐到 num_steps
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 给源序列添加 batch 维度，形状从 (num_steps,) 变成 (1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # 编码器对源句子编码，得到源句子每个 token 的编码结果，enc_outputs 的形状是 (1, num_steps, num_hiddens)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 用编码器输出初始化解码器状态，
    # dec_state 一般包含：[enc_outputs, enc_valid_lens, [None]*num_layers]，
    # 其中 enc_outputs 是编码器的输出表示，enc_valid_lens 是编码器的有效长度，第三个元素是一个长度为 num_layers 的列表，用于缓存每一层 DecoderBlock 解码的历史输出表示
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 构造解码器初始输入 <bos> 的 batch，形状是 (1, 1)，表示 batch_size=1，当前输入长度=1
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    # output_seq：保存预测出来的 token index
    # attention_weight_seq：保存每一步的注意力权重
    output_seq, attention_weight_seq = [], []
    # 循环生成目标 token，最多生成 num_steps 个 token。如果模型一直不预测 <eos>，循环就会无限进行
    for _ in range(num_steps):
        # 解码器预测下一个 token。
        # 第一次循环时：dec_X = [[<bos>]]，dec_state 包含编码器输出和有效长度；
        # 之后的循环时：dec_X = [[pred]]，dec_state 包含编码器输出、有效长度和之前解码的输出表示
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 选择概率最大的 token，作为下一时刻解码器输入。
        # Y 是 logits，形状是 (1, 1, vocab_size)；
        # 沿着 dim=2 取最大值的位置，也就是在词表维度上选分数最高的 token，
        # 得到 dec_X 的形状是 (1, 1)，里面是预测 token 的 index
        dec_X = Y.argmax(dim=2)
        # 取出预测 token 的整数编号，等价于 dec_X[0, 0].item()
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 如果需要，保存注意力权重，用于后续可视化
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 如果预测到 <eos>，说明句子生成结束，不再继续生成
        if pred == tgt_vocab['<eos>']:
            break
        # 保存当前预测 token 的 index
        output_seq.append(pred)
    # 把预测出的 token index 转回 token，并拼成句子返回
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算 BLEU 分数。

    参数:
        pred_seq: 模型预测出的句子，字符串形式，例如 'the cat is on the mat'。
        label_seq: 真实参考句子，字符串形式，例如 'the cat sat on the mat'。
        k: 最大 n-gram 阶数。例如 k=2 表示同时考虑 1-gram 和 2-gram。

    返回:
        score: BLEU 分数，范围通常在 0 到 1 之间，越接近 1 表示预测句子越接近参考句子。
    """

    # 按空格分词
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')

    # 获取预测序列长度和标签序列长度
    len_pred, len_label = len(pred_tokens), len(label_tokens)

    # 长度惩罚项：如果预测句子太短，会被惩罚
    score = math.exp(min(0, 1 - len_label / len_pred))

    # 依次计算 1-gram, 2-gram, ..., k-gram 的匹配情况
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)

        # 统计真实句子中所有 n-gram 的出现次数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1

        # 遍历预测句子的所有 n-gram，统计有多少能在真实句子中匹配到
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                # 截断计数 clipped count，防止重复匹配过多
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1

        # 将当前 n-gram 的精确率乘到总分中，score *= 当前 n-gram 精确率 ^ 权重
        score *= math.pow(
            num_matches / (len_pred - n + 1),
            math.pow(0.5, n))

    return score

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {bleu(translation, fra, k=2):.3f}')