import torch
from torch import nn
from transformer import EncoderBlock


class BERTEncoder(nn.Module):
    """BERT 编码器。

    BERTEncoder 的作用是把输入 token 序列编码成上下文表示。

    BERT 的输入表示由三部分相加得到:
        1. token_embedding: 词元嵌入，表示 token 本身是什么；
        2. segment_embedding: 片段嵌入，表示 token 属于句子 A 还是句子 B；
        3. pos_embedding: 位置嵌入，表示 token 在序列中的位置。

    然后将三者相加后的结果输入多个 Transformer EncoderBlock。
    """

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        """初始化 BERTEncoder。

        参数:
            vocab_size: 词表大小。
                例如词表中有 10000 个 token，则 vocab_size=10000。

            num_hiddens: 隐藏层维度，也是 token embedding、segment embedding、
                position embedding 的维度。
                BERT-base 中 num_hiddens=768。

            norm_shape: LayerNorm 的归一化维度。
                通常为 [num_hiddens]。
                例如 BERT-base 中通常是 [768]。

            ffn_num_input: 前馈网络 FFN 的输入维度。
                通常等于 num_hiddens。

            ffn_num_hiddens: 前馈网络 FFN 的隐藏层维度。
                BERT-base 中通常是 3072，即 768 的 4 倍。

            num_heads: 多头注意力的头数。
                BERT-base 中 num_heads=12。

            num_layers: Transformer EncoderBlock 的层数。
                BERT-base 中 num_layers=12。

            dropout: Dropout 概率。
                BERT-base 中通常为 0.1。

            max_len: 最大序列长度。
                用于创建可学习的位置嵌入矩阵。
                BERT-base 中通常是 512，这里默认是 1000。

            key_size: key 的输入维度。
                通常应与 num_hiddens 一致。
                BERT-base 中 key_size=768。

            query_size: query 的输入维度。
                通常应与 num_hiddens 一致。
                BERT-base 中 query_size=768。

            value_size: value 的输入维度。
                通常应与 num_hiddens 一致。
                BERT-base 中 value_size=768。

            **kwargs: 其他传给 nn.Module 的参数。
        """

        super(BERTEncoder, self).__init__(**kwargs)

        # token embedding:
        # 把 token id 转换成 num_hiddens 维向量
        # 输入 tokens shape: (batch_size, num_steps)
        # 输出 shape: (batch_size, num_steps, num_hiddens)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)

        # segment embedding:
        # 用于区分句子 A 和句子 B
        # segment id 只有 0 和 1 两种，所以 nn.Embedding(2, num_hiddens)
        # 输入 segments shape: (batch_size, num_steps)
        # 输出 shape: (batch_size, num_steps, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)

        # 创建多个 Transformer EncoderBlock
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    True
                )
            )

        # 位置嵌入 position embedding:
        # BERT 中使用的是可学习的绝对位置嵌入，不是固定 sin/cos 位置编码。
        #
        # self.pos_embedding shape:
        # (1, max_len, num_hiddens)
        #
        # 第 0 维为 1，是为了后续和 batch 维度广播相加。
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, num_hiddens)
        )

    def forward(self, tokens, segments, valid_lens):
        """执行 BERTEncoder 前向传播。

        参数:
            tokens: 输入 token id。
                shape 为 (batch_size, num_steps)。

            segments: 片段索引。
                shape 为 (batch_size, num_steps)。
                其中 0 表示句子 A，1 表示句子 B。

            valid_lens: 每条序列的有效长度。
                shape 通常为 (batch_size,)。
                用于在注意力中屏蔽 padding 位置。

        返回:
            X: BERT 编码后的上下文表示。
                shape 为 (batch_size, num_steps, num_hiddens)。
        """

        # 1. token embedding + segment embedding
        #
        # token_embedding(tokens) shape:
        # (batch_size, num_steps, num_hiddens)
        #
        # segment_embedding(segments) shape:
        # (batch_size, num_steps, num_hiddens)
        #
        # 两者相加后 X shape 仍然是:
        # (batch_size, num_steps, num_hiddens)
        X = self.token_embedding(tokens) + \
            self.segment_embedding(segments)

        # 2. 加上位置嵌入
        #
        # self.pos_embedding[:, :X.shape[1], :] shape:
        # (1, num_steps, num_hiddens)
        #
        # 通过广播机制与 X 相加:
        # (batch_size, num_steps, num_hiddens)
        # +
        # (1, num_steps, num_hiddens)
        #
        # 得到:
        # (batch_size, num_steps, num_hiddens)
        #
        # 注意:
        # 如果希望位置嵌入参与训练，建议不要使用 .data。
        # 更推荐写成:
        X = X + self.pos_embedding[:, :X.shape[1], :]
        # X = X + self.pos_embedding.data[:, :X.shape[1], :]

        # 3. 依次通过多个 Transformer EncoderBlock
        #
        # 每个 blk 都会执行:
        # Multi-Head Self-Attention
        # AddNorm
        # FFN
        # AddNorm
        #
        # valid_lens 用于屏蔽 padding 位置。
        for blk in self.blks:
            X = blk(X, valid_lens)

        # 返回每个 token 的上下文表示
        return X


class MaskLM(nn.Module):
    """BERT 的掩蔽语言模型任务 Masked Language Model, MLM。

    作用:
        从 BERTEncoder 的输出 X 中取出被 mask 的位置对应的隐藏表示，
        然后通过一个 MLP 预测这些位置原本应该是什么 token。

    输入:
        X:
            BERTEncoder 的输出。
            shape: (batch_size, num_steps, num_inputs)
            其中 num_inputs 通常等于 BERT 的隐藏维度，例如 768。

        pred_positions:
            需要预测的 mask 位置索引。
            shape: (batch_size, num_pred_positions)
            每一行表示当前样本中哪些位置需要做 MLM 预测。

    输出:
        mlm_Y_hat:
            被 mask 位置的预测结果 logits。
            shape: (batch_size, num_pred_positions, vocab_size)
            每个 mask 位置都会输出一个 vocab_size 维向量，
            表示该位置预测为词表中每个 token 的分数。
    """

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        """初始化 MLM 预测头。

        参数:
            vocab_size: 词表大小。
                MLM 最终要在整个词表上做分类，所以输出维度是 vocab_size。

            num_hiddens: MLP 隐藏层维度。
                用于对 mask 位置的隐藏表示做进一步变换。

            num_inputs: 输入隐藏表示的维度。
                通常等于 BERTEncoder 的输出维度。
                BERT-base 中一般是 768。

            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(MaskLM, self).__init__(**kwargs)

        # MLP 预测头:
        # num_inputs -> num_hiddens -> vocab_size
        #
        # 对每个被 mask 的位置，输出一个 vocab_size 维 logits，
        # 用于预测该位置原来的 token。
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        """执行 MLM 前向传播。

        参数:
            X: BERTEncoder 输出。
                shape: (batch_size, num_steps, num_inputs)

            pred_positions: 每个样本中需要预测的 token 位置。
                shape: (batch_size, num_pred_positions)

        返回:
            mlm_Y_hat: mask 位置的预测 logits。
                shape: (batch_size, num_pred_positions, vocab_size)
        """

        # 每个样本中需要预测的位置数量
        # 例如 pred_positions.shape = (2, 3)，则 num_pred_positions = 3
        num_pred_positions = pred_positions.shape[1]

        # 将 pred_positions 拉平成一维
        # 原 shape: (batch_size, num_pred_positions)
        # 新 shape: (batch_size * num_pred_positions,)
        #
        # 例如:
        # [[1, 4, 6],
        #  [2, 3, 5]]
        # 变成:
        # [1, 4, 6, 2, 3, 5]
        pred_positions = pred_positions.reshape(-1)

        # 获取 batch_size
        batch_size = X.shape[0]

        # 构造 batch 索引
        # 假设 batch_size = 2
        # 初始 batch_idx = [0, 1]
        #
        # 注意:
        # 如果 X 在 GPU 上，batch_idx 也应该放到 X.device 上，
        # 否则可能出现 CPU/GPU 设备不一致错误。
        batch_idx = torch.arange(0, batch_size)

        # 将每个 batch 索引重复 num_pred_positions 次
        #
        # 假设 batch_size = 2，num_pred_positions = 3
        # 原 batch_idx:
        # [0, 1]
        #
        # repeat_interleave 后:
        # [0, 0, 0, 1, 1, 1]
        #
        # 它会和 pred_positions 一一对应，用于从 X 中取出 mask 位置的隐藏表示。
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        # 根据 batch_idx 和 pred_positions 从 X 中取出被 mask 位置的隐藏表示
        #
        # X shape:
        # (batch_size, num_steps, num_inputs)
        #
        # batch_idx shape:
        # (batch_size * num_pred_positions,)
        #
        # pred_positions shape:
        # (batch_size * num_pred_positions,)
        #
        # masked_X shape:
        # (batch_size * num_pred_positions, num_inputs)
        #
        # 例如会取:
        # X[0, 1], X[0, 4], X[0, 6],
        # X[1, 2], X[1, 3], X[1, 5]
        masked_X = X[batch_idx, pred_positions]

        # 恢复成三维:
        # (batch_size, num_pred_positions, num_inputs)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))

        # 对每个 mask 位置做词表分类预测
        #
        # 输入 masked_X shape:
        # (batch_size, num_pred_positions, num_inputs)
        #
        # 输出 mlm_Y_hat shape:
        # (batch_size, num_pred_positions, vocab_size)
        mlm_Y_hat = self.mlp(masked_X)

        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """BERT 的下一句预测任务 NSP, Next Sentence Prediction。

    作用:
        根据 BERTEncoder 输出中 <cls> 位置的隐藏表示，
        判断输入的两个句子是否具有“下一句”关系。

    输入:
        X:
            通常是 <cls> 位置对应的隐藏表示。
            shape: (batch_size, num_inputs)

    输出:
        nsp_Y_hat:
            下一句预测 logits。
            shape: (batch_size, 2)

            其中第 0 类通常表示:
                句子 B 是句子 A 的下一句

            第 1 类通常表示:
                句子 B 不是句子 A 的下一句
    """

    def __init__(self, num_inputs, **kwargs):
        """初始化 NSP 预测头。

        参数:
            num_inputs: 输入特征维度。
                通常等于 BERTEncoder 的隐藏维度 num_hiddens。
                例如 BERT-base 中 num_inputs = 768。

            **kwargs: 其他传给 nn.Module 的参数。
        """
        super(NextSentencePred, self).__init__(**kwargs)

        # NSP 是一个二分类任务，所以输出维度是 2
        # 输入:  (batch_size, num_inputs)
        # 输出:  (batch_size, 2)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """执行下一句预测。

        参数:
            X: <cls> 位置的隐藏表示。
                shape: (batch_size, num_inputs)

        返回:
            NSP 分类 logits。
            shape: (batch_size, 2)
        """

        # 对 <cls> 表示做线性分类，输出两个类别的 logits
        return self.output(X)


class BERTModel(nn.Module):
    """BERT 模型。

    该模型由三部分组成：
        1. BERTEncoder：对输入 token 序列进行编码，输出每个 token 的上下文表示；
        2. MaskLM：掩蔽语言模型预测头，用于预测被 mask 位置的原始 token；
        3. NextSentencePred：下一句预测头，用于判断句子 B 是否是句子 A 的下一句。

    输入：
        tokens:
            token id 序列，shape 为 (batch_size, num_steps)。

        segments:
            片段编号序列，shape 为 (batch_size, num_steps)。
            0 表示句子 A，1 表示句子 B。

        valid_lens:
            每条样本的有效长度，shape 通常为 (batch_size,)。
            用于在 self-attention 中屏蔽 <pad> 位置。

        pred_positions:
            MLM 任务中需要预测的位置，shape 为 (batch_size, num_pred_positions)。
            如果为 None，则不执行 MLM 预测。

    输出：
        encoded_X:
            BERTEncoder 输出的上下文表示。
            shape 为 (batch_size, num_steps, num_hiddens)。

        mlm_Y_hat:
            MLM 预测结果。
            如果 pred_positions 不为 None，
            shape 为 (batch_size, num_pred_positions, vocab_size)；
            否则为 None。

        nsp_Y_hat:
            NSP 预测结果。
            shape 为 (batch_size, 2)。
    """

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        """初始化 BERTModel。

        参数:
            vocab_size:
                词表大小。MLM 最终需要预测词表中的 token，所以会用到 vocab_size。

            num_hiddens:
                BERT 隐藏层维度。
                BERT-base 中通常是 768。

            norm_shape:
                LayerNorm 的归一化维度。
                通常为 [num_hiddens]，例如 [768]。

            ffn_num_input:
                Transformer EncoderBlock 中 FFN 的输入维度。
                通常等于 num_hiddens。

            ffn_num_hiddens:
                Transformer EncoderBlock 中 FFN 的隐藏层维度。
                BERT-base 中通常是 3072。

            num_heads:
                多头注意力头数。
                BERT-base 中通常是 12。

            num_layers:
                Transformer EncoderBlock 层数。
                BERT-base 中通常是 12。

            dropout:
                Dropout 概率。
                BERT-base 中通常是 0.1。

            max_len:
                最大位置嵌入长度。
                标准 BERT-base 中通常是 512。
                这里默认是 1000。

            key_size:
                key 输入维度。
                通常等于 num_hiddens。

            query_size:
                query 输入维度。
                通常等于 num_hiddens。

            value_size:
                value 输入维度。
                通常等于 num_hiddens。

            hid_in_features:
                NSP 前置隐藏层的输入维度。
                通常等于 num_hiddens。

            mlm_in_features:
                MaskLM 输入特征维度。
                通常等于 num_hiddens。

            nsp_in_features:
                NextSentencePred 输入特征维度。
                通常等于 num_hiddens。
        """

        super(BERTModel, self).__init__()

        # BERT 编码器主体
        # 输入 tokens、segments、valid_lens
        # 输出 encoded_X，shape 为:
        # (batch_size, num_steps, num_hiddens)
        self.encoder = BERTEncoder(
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len=max_len,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size
        )

        # NSP 任务使用的 <cls> 表示变换层
        #
        # encoded_X[:, 0, :] 是 <cls> 位置的输出表示，
        # shape 为 (batch_size, hid_in_features)。
        #
        # 经过 hidden 后 shape 仍然是:
        # (batch_size, num_hiddens)
        #
        # BERT 原始实现里，pooler 通常是 Linear + Tanh。
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh()
        )

        # MLM 预测头
        # 用于预测被 mask 位置原来的 token
        #
        # 输入:
        # encoded_X 和 pred_positions
        #
        # 输出:
        # mlm_Y_hat shape:
        # (batch_size, num_pred_positions, vocab_size)
        self.mlm = MaskLM(
            vocab_size,
            num_hiddens,
            mlm_in_features
        )

        # NSP 预测头
        # 用于判断句子 B 是否是句子 A 的下一句
        #
        # 输入:
        # <cls> 表示经过 hidden 后的结果
        #
        # 输出:
        # nsp_Y_hat shape:
        # (batch_size, 2)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        """执行 BERTModel 前向传播。

        参数:
            tokens:
                token id 序列。
                shape 为 (batch_size, num_steps)。

            segments:
                segment id 序列。
                shape 为 (batch_size, num_steps)。

            valid_lens:
                每条序列的有效长度。
                shape 通常为 (batch_size,)。
                用于屏蔽 <pad>。

            pred_positions:
                MLM 任务中需要预测的位置。
                shape 为 (batch_size, num_pred_positions)。
                如果为 None，则不计算 MLM 预测结果。

        返回:
            encoded_X:
                BERTEncoder 输出。
                shape 为 (batch_size, num_steps, num_hiddens)。

            mlm_Y_hat:
                MLM 预测 logits。
                shape 为 (batch_size, num_pred_positions, vocab_size)。
                如果 pred_positions 为 None，则返回 None。

            nsp_Y_hat:
                NSP 预测 logits。
                shape 为 (batch_size, 2)。
        """

        # 1. 通过 BERTEncoder 编码输入序列
        #
        # tokens shape:
        # (batch_size, num_steps)
        #
        # segments shape:
        # (batch_size, num_steps)
        #
        # encoded_X shape:
        # (batch_size, num_steps, num_hiddens)
        encoded_X = self.encoder(tokens, segments, valid_lens)

        # 2. 如果给出了 pred_positions，就执行 MLM 预测
        #
        # pred_positions shape:
        # (batch_size, num_pred_positions)
        #
        # mlm_Y_hat shape:
        # (batch_size, num_pred_positions, vocab_size)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            # 如果没有提供 pred_positions，说明当前不需要做 MLM 任务
            mlm_Y_hat = None

        # 3. NSP 下一句预测
        #
        # encoded_X[:, 0, :] 取的是每条样本中第 0 个位置的表示，
        # 第 0 个位置通常是 <cls>。
        #
        # encoded_X[:, 0, :] shape:
        # (batch_size, num_hiddens)
        #
        # self.hidden(...) shape:
        # (batch_size, num_hiddens)
        #
        # nsp_Y_hat shape:
        # (batch_size, 2)
        nsp_Y_hat = self.nsp(
            self.hidden(encoded_X[:, 0, :])
        )

        return encoded_X, mlm_Y_hat, nsp_Y_hat
