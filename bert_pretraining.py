import torch
from torch import nn
from transformer import Timer, Accumulator
from bert import BERTModel
from bert_dataset import load_data_wiki, _get_tokens_and_segments


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    """计算一个 batch 的 BERT 预训练损失。

    参数:
        net:
            BERTModel 模型。

        loss:
            交叉熵损失函数。
            推荐定义为:
                nn.CrossEntropyLoss(reduction='none')
            因为 MLM 任务需要先得到每个预测位置的 loss，
            再用 mlm_weights_X 过滤掉 padding 出来的无效预测位置。

        vocab_size:
            词表大小。
            MLM 任务最终要预测词表中的 token，所以输出维度是 vocab_size。

        tokens_X:
            输入 token id。
            shape: (batch_size, max_len)

        segments_X:
            segment id。
            shape: (batch_size, max_len)
            0 表示句子 A，1 表示句子 B。

        valid_lens_x:
            每条样本的有效长度，不包括 <pad>。
            shape: (batch_size,)

        pred_positions_X:
            MLM 任务中需要预测的位置。
            shape: (batch_size, max_num_mlm_preds)

        mlm_weights_X:
            MLM loss 权重。
            真实预测位置为 1，padding 出来的预测位置为 0。
            shape: (batch_size, max_num_mlm_preds)

        mlm_Y:
            MLM 真实标签。
            shape: (batch_size, max_num_mlm_preds)

        nsp_y:
            NSP 真实标签。
            shape: (batch_size,)

    返回:
        mlm_l:
            当前 batch 的 MLM 平均损失。

        nsp_l:
            当前 batch 的 NSP 平均损失。

        l:
            总损失，等于 mlm_l + nsp_l。
    """

    # =========================
    # 1. 前向传播
    # =========================
    #
    # net 返回三个值:
    # encoded_X:
    #   BERTEncoder 的输出，shape: (batch_size, max_len, num_hiddens)
    #
    # mlm_Y_hat:
    #   MLM 预测结果，shape: (batch_size, max_num_mlm_preds, vocab_size)
    #
    # nsp_Y_hat:
    #   NSP 预测结果，shape: (batch_size, 2)
    #
    # 这里第一个 encoded_X 暂时不用，所以用 _ 接收。
    _, mlm_Y_hat, nsp_Y_hat = net(
        tokens_X,
        segments_X,
        valid_lens_x.reshape(-1),
        pred_positions_X
    )

    # =========================
    # 2. 计算 MLM loss
    # =========================
    #
    # mlm_Y_hat 原始形状:
    # (batch_size, max_num_mlm_preds, vocab_size)
    #
    # CrossEntropyLoss 要求输入形状为:
    # (样本数, 类别数)
    #
    # 所以 reshape 成:
    # (batch_size * max_num_mlm_preds, vocab_size)
    #
    # mlm_Y 原始形状:
    # (batch_size, max_num_mlm_preds)
    #
    # reshape 成:
    # (batch_size * max_num_mlm_preds,)
    mlm_l = loss(
        mlm_Y_hat.reshape(-1, vocab_size),
        mlm_Y.reshape(-1)
    )

    # 如果 loss = nn.CrossEntropyLoss(reduction='none')
    # 那么 mlm_l 的形状是:
    # (batch_size * max_num_mlm_preds,)
    #
    # mlm_weights_X 原 shape:
    # (batch_size, max_num_mlm_preds)
    #
    # 所以 reshape 成:
    # (batch_size * max_num_mlm_preds,)
    #
    # 真实 MLM 预测位置权重为 1；
    # padding 出来的预测位置权重为 0。
    mlm_weights = mlm_weights_X.reshape(-1)

    # 用权重过滤掉 padding 出来的无效 MLM 预测位置
    #
    # 例如:
    # mlm_l       = [2.3, 1.5, 4.1, 0.7]
    # mlm_weights = [1.0, 0.0, 1.0, 0.0]
    #
    # 加权后:
    # [2.3, 0.0, 4.1, 0.0]
    mlm_l = mlm_l * mlm_weights

    # 只对真实 MLM 预测位置求平均
    #
    # mlm_weights_X.sum() 表示当前 batch 中真实 MLM 预测位置的数量。
    # 加 1e-8 是为了防止极端情况下除以 0。
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    # =========================
    # 3. 计算 NSP loss
    # =========================
    #
    # nsp_Y_hat shape:
    # (batch_size, 2)
    #
    # nsp_y shape:
    # (batch_size,)
    #
    # 如果 loss 使用 reduction='none'，
    # 那么 nsp_l 的形状是:
    # (batch_size,)
    #
    # 所以这里手动 mean。
    nsp_l = loss(nsp_Y_hat, nsp_y)
    nsp_l = nsp_l.mean()

    # =========================
    # 4. 总 loss
    # =========================
    #
    # BERT 预训练总损失 = MLM 损失 + NSP 损失
    l = mlm_l + nsp_l

    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    """训练 BERT 模型。

    参数:
        train_iter:
            BERT 预训练数据迭代器。
            每次返回:
                tokens_X,
                segments_X,
                valid_lens_x,
                pred_positions_X,
                mlm_weights_X,
                mlm_Y,
                nsp_y

        net:
            BERTModel 模型。

        loss:
            交叉熵损失函数。
            推荐:
                nn.CrossEntropyLoss(reduction='none')

        vocab_size:
            词表大小。

        devices:
            训练设备列表。
            例如:
                [torch.device('cuda:0')]
                或
                [torch.device('cpu')]

        num_steps:
            总共训练多少个 step。
    """

    # =========================
    # 1. 模型放到设备上
    # =========================
    #
    # nn.DataParallel 用于多 GPU 并行训练。
    # 如果 devices 有多个 GPU，比如 [cuda:0, cuda:1]，
    # 它会把 batch 切分到多个 GPU 上计算。
    #
    # devices[0] 是主设备。
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # =========================
    # 2. 定义优化器
    # =========================
    #
    # Adam 优化器。
    # 注意：lr=0.01 对 BERT 来说通常偏大，
    # D2L 教学代码里模型较小，所以可以这样演示。
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)

    # step: 当前已经训练了多少步
    # timer: 用于统计训练速度
    step, timer = 0, Timer()

    # =========================
    # 3. 定义统计器
    # =========================
    #
    # metric 用来累计 4 个值:
    #
    # metric[0]: MLM loss 累加值
    # metric[1]: NSP loss 累加值
    # metric[2]: 已处理的句子对数量
    # metric[3]: 已训练的 step 数量
    metric = Accumulator(4)

    # 用于控制达到 num_steps 后跳出双层循环
    num_steps_reached = False

    # =========================
    # 4. 开始训练循环
    # =========================
    #
    # 外层 while:
    #   保证总 step 数达到 num_steps。
    #
    # 内层 for:
    #   遍历 train_iter 中的 batch。
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:

            # =========================
            # 4.1 把 batch 数据放到训练设备
            # =========================

            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            nsp_y = nsp_y.to(devices[0])

            # =========================
            # 4.2 梯度清零
            # =========================
            #
            # PyTorch 默认会累积梯度。
            # 所以每个 batch 反向传播前要先清空上一轮梯度。
            trainer.zero_grad()
            
            # 开始计时
            timer.start()

            # =========================
            # 4.3 前向传播 + 计算 loss
            # =========================
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net,
                loss,
                vocab_size,
                tokens_X,
                segments_X,
                valid_lens_x,
                pred_positions_X,
                mlm_weights_X,
                mlm_Y,
                nsp_y
            )

            # =========================
            # 4.4 反向传播
            # =========================
            #
            # 根据总损失 l 计算所有可学习参数的梯度。
            l.backward()

            # =========================
            # 4.5 更新参数
            # =========================
            #
            # Adam 根据梯度更新模型参数。
            trainer.step()

            # =========================
            # 4.6 累计统计指标
            # =========================
            #
            # mlm_l 和 nsp_l 是 tensor。
            # 为避免把计算图保存进 Accumulator，
            # 建议转成普通 Python 数值。
            metric.add(
                mlm_l.detach().cpu().item(),
                nsp_l.detach().cpu().item(),
                tokens_X.shape[0],
                1
            )

            # 停止计时
            timer.stop()

            # 当前训练 step 加 1
            step += 1

            # =========================
            # 4.7 打印训练过程
            # =========================
            #
            # metric[0] / metric[3]:
            #   从训练开始到当前 step 的平均 MLM loss
            #
            # metric[1] / metric[3]:
            #   从训练开始到当前 step 的平均 NSP loss
            print(
                f'step {step:4d}/{num_steps}, '
                f'MLM loss {metric[0] / metric[3]:.4f}, '
                f'NSP loss {metric[1] / metric[3]:.4f}'
            )

            # 如果达到指定训练步数，就跳出内层 for 循环
            if step == num_steps:
                num_steps_reached = True
                break

    # =========================
    # 5. 打印最终结果
    # =========================

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')

    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


def try_all_gpus():
    """返回当前机器上所有可用的 GPU 设备；如果没有 GPU，则返回 CPU。

    返回:
        devices: 一个 torch.device 对象组成的列表。

        如果有 GPU，例如有 2 张 GPU:
            [device(type='cuda', index=0), device(type='cuda', index=1)]

        如果没有 GPU:
            [device(type='cpu')]
    """

    # torch.cuda.device_count() 会返回当前 PyTorch 能检测到的 GPU 数量
    #
    # 例如:
    #   如果有 2 张 GPU，则 torch.cuda.device_count() = 2
    #   range(2) = [0, 1]
    #
    # 然后构造:
    #   torch.device('cuda:0')
    #   torch.device('cuda:1')
    devices = [
        torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())
    ]

    # 如果 devices 不为空，说明至少检测到一张 GPU，直接返回所有 GPU
    # 如果 devices 为空，说明没有检测到 GPU，则返回 CPU
    return devices if devices else [torch.device('cpu')]


if __name__ == "__main__":

    # =========================
    # 1. 设置数据加载参数
    # =========================

    # batch_size:
    #   每个 batch 中包含 512 条训练样本。
    #
    # max_len:
    #   每条 BERT 输入序列的最大长度为 64。
    #   如果样本长度不足 64，会补 <pad>；
    #   如果构造出的句子对长度超过 64，会在前面的数据构造阶段跳过。
    batch_size, max_len = 512, 64

    # 加载 WikiText-2 数据集
    #
    # train_iter:
    #   BERT 预训练数据迭代器。
    #   每次返回一个 batch，包括：
    #       tokens_X,
    #       segments_X,
    #       valid_lens_x,
    #       pred_positions_X,
    #       mlm_weights_X,
    #       mlm_Y,
    #       nsp_y
    #
    # vocab:
    #   根据 WikiText-2 训练集构建出的词表。
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    # =========================
    # 2. 构造 BERT 模型
    # =========================

    # 这里构造的不是标准 BERT-base，而是一个小型 BERT。
    #
    # 标准 BERT-base 一般是：
    #   hidden_size = 768
    #   num_layers = 12
    #   num_heads = 12
    #   ffn_num_hiddens = 3072
    #
    # 这里为了教学和快速训练，设置成：
    #   hidden_size = 128
    #   num_layers = 2
    #   num_heads = 2
    #   ffn_num_hiddens = 256
    net = BERTModel(
        # 词表大小。
        # MLM 任务最终要在整个词表上预测 token，
        # 所以 vocab_size = len(vocab)。
        len(vocab),

        # BERT 隐藏层维度。
        # 每个 token 最终会表示成 128 维向量。
        num_hiddens=128,

        # LayerNorm 的归一化维度。
        # 因为隐藏层维度是 128，所以这里是 [128]。
        norm_shape=[128],

        # FFN 前馈网络输入维度。
        # 通常等于 num_hiddens。
        ffn_num_input=128,

        # FFN 前馈网络隐藏层维度。
        # 小型模型中设置为 256。
        # 对应结构是：128 -> 256 -> 128。
        ffn_num_hiddens=256,

        # 多头注意力头数。
        # num_hiddens=128，num_heads=2，
        # 所以每个 head 的维度是 128 / 2 = 64。
        num_heads=2,

        # Transformer EncoderBlock 层数。
        # 这里堆叠 2 层 EncoderBlock。
        num_layers=2,

        # dropout 概率。
        # 训练时随机丢弃一部分神经元，防止过拟合。
        dropout=0.2,

        # Q、K、V 的输入维度。
        # 通常都等于 num_hiddens。
        key_size=128,
        query_size=128,
        value_size=128,

        # hidden 层输入维度。
        # 主要用于 NSP 任务中 <cls> 表示的变换。
        hid_in_features=128,

        # MLM 预测头输入维度。
        mlm_in_features=128,

        # NSP 预测头输入维度。
        nsp_in_features=128
    )
    print(net)

    # =========================
    # 3. 选择训练设备
    # =========================

    # try_all_gpus 会返回所有可用 GPU。
    # 如果没有 GPU，则返回 [torch.device('cpu')]。
    #
    # 例如：
    #   有 1 张 GPU: [cuda:0]
    #   没有 GPU:   [cpu]
    devices = try_all_gpus()

    # =========================
    # 4. 定义损失函数
    # =========================

    # reduction='none' 表示不直接求平均，
    # 而是保留每个样本 / 每个预测位置的 loss。
    #
    # 这是必要的，因为 MLM 中有些 pred_positions 是 padding 出来的，
    # 需要用 mlm_weights 把这些无效位置过滤掉。
    loss = nn.CrossEntropyLoss(reduction='none')

    # =========================
    # 5. 训练 BERT
    # =========================

    # 训练 50 个 step。
    #
    # 注意：
    #   这里只是教学演示，50 step 训练出来的模型还很弱，
    #   不能期待它学到非常好的语言表示。
    train_bert(train_iter, net, loss, len(vocab), devices, 50)

    def get_bert_encoding(net, tokens_a, tokens_b=None):
        """使用训练后的 BERT 对单句或句子对进行编码。

        参数:
            net:
                训练后的 BERTModel。

            tokens_a:
                句子 A 的 token 列表。
                例如:
                    ['a', 'crane', 'is', 'flying']

            tokens_b:
                句子 B 的 token 列表，默认为 None。
                如果不为 None，则构造句子对输入。

        返回:
            encoded_X:
                BERTEncoder 输出的上下文表示。
                shape:
                    (1, 序列长度, num_hiddens)
        """

        # 构造 BERT 输入 token 和 segment id。
        #
        # 如果只有 tokens_a:
        #   tokens:
        #       <cls> tokens_a <sep>
        #
        # 如果有 tokens_a 和 tokens_b:
        #   tokens:
        #       <cls> tokens_a <sep> tokens_b <sep>
        #
        # segments:
        #   句子 A 部分为 0；
        #   句子 B 部分为 1。
        tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)

        # 将 token 转成 token id，并添加 batch 维度。
        #
        # vocab[tokens] 得到一维列表：
        #   shape: (seq_len,)
        #
        # unsqueeze(0) 后：
        #   shape: (1, seq_len)
        token_ids = torch.tensor(
            vocab[tokens],
            device=devices[0]
        ).unsqueeze(0)

        # 将 segment id 转成 tensor，并添加 batch 维度。
        #
        # shape:
        #   (1, seq_len)
        segments = torch.tensor(
            segments,
            device=devices[0]
        ).unsqueeze(0)

        # valid_len 表示当前输入的有效长度。
        # 这里没有 padding，所以有效长度就是 len(tokens)。
        #
        # shape:
        #   (1,)
        valid_len = torch.tensor(
            len(tokens),
            device=devices[0]
        ).unsqueeze(0)

        # 前向传播。
        #
        # 因为这里 pred_positions=None，
        # 所以不会执行 MLM 预测，mlm_Y_hat 会是 None。
        #
        # encoded_X shape:
        #   (1, seq_len, num_hiddens)
        encoded_X, _, _ = net(token_ids, segments, valid_len)

        return encoded_X

    # =========================
    # 6. 编码单句
    # =========================

    # 单句:
    #   a crane is flying
    #
    # 注意：
    #   crane 这里更可能表示“鹤”。
    tokens_a = ['a', 'crane', 'is', 'flying']
    
    encoded_text = get_bert_encoding(net, tokens_a)
    
    # 实际输入 token 是：
    #   '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
    #
    # 所以序列长度是：
    #   6
    #
    # encoded_text shape:
    #   (1, 6, 128)
    encoded_text_cls = encoded_text[:, 0, :]
    
    # crane 在序列中的位置是 2：
    #   0: <cls>
    #   1: a
    #   2: crane
    #   3: is
    #   4: flying
    #   5: <sep>
    encoded_text_crane = encoded_text[:, 2, :]
    
    # 打印：
    #   encoded_text.shape:
    #       整个句子的编码结果形状
    #
    #   encoded_text_cls.shape:
    #       <cls> 位置表示的形状
    #
    #   encoded_text_crane[0][:3]:
    #       crane 这个词向量的前 3 个维度
    print(
        encoded_text.shape,
        encoded_text_cls.shape,
        encoded_text_crane[0][:3]
    )

    # =========================
    # 7. 编码句子对
    # =========================

    # 句子 A:
    #   a crane driver came
    #
    # 句子 B:
    #   he just left
    #
    # 注意：
    #   crane 这里更可能表示“起重机”。
    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']

    encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)

    # 实际输入 token 是：
    #   '<cls>', 'a', 'crane', 'driver', 'came', '<sep>',
    #   'he', 'just', 'left', '<sep>'
    #
    # 所以序列长度是：
    #   10
    #
    # encoded_pair shape:
    #   (1, 10, 128)
    encoded_pair_cls = encoded_pair[:, 0, :]

    # crane 在句子对输入中的位置仍然是 2：
    #   0: <cls>
    #   1: a
    #   2: crane
    #   3: driver
    #   4: came
    #   5: <sep>
    #   6: he
    #   7: just
    #   8: left
    #   9: <sep>
    encoded_pair_crane = encoded_pair[:, 2, :]

    # 打印：
    #   encoded_pair.shape:
    #       句子对编码结果形状
    #
    #   encoded_pair_cls.shape:
    #       <cls> 位置表示形状
    #
    #   encoded_pair_crane[0][:3]:
    #       句子对中 crane 这个词的上下文表示前 3 个维度
    print(
        encoded_pair.shape,
        encoded_pair_cls.shape,
        encoded_pair_crane[0][:3]
    )
