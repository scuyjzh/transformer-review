import multiprocessing
import os
import re
import json
import torch
from torch import nn
from bert import BERTModel
from bert_dataset import tokenize, _get_tokens_and_segments
from bert_pretraining import try_all_gpus
from transformer import Vocab, Timer, Accumulator


class SNLIBERTDataset(torch.utils.data.Dataset):
    """用于将 SNLI 数据集处理成 BERT 微调所需格式的 Dataset。

    每条样本最终返回:
        ((token_ids, segments, valid_len), label)

    其中:
        token_ids:
            BERT 输入 token id，shape: (max_len,)

        segments:
            BERT segment id，shape: (max_len,)
            0 表示 premise 部分，1 表示 hypothesis 部分。

        valid_len:
            当前样本的有效长度，不包含 <pad>。

        label:
            SNLI 三分类标签:
                entailment     -> 0
                contradiction  -> 1
                neutral        -> 2
    """

    def __init__(self, dataset, max_len, vocab=None):
        """初始化 SNLIBERTDataset。

        参数:
            dataset:
                一个三元组:
                    dataset[0]: premises，前提句列表
                    dataset[1]: hypotheses，假设句列表
                    dataset[2]: labels，标签列表

                例如:
                    dataset = (
                        ['A dog is running'],
                        ['An animal runs'],
                        [0]
                    )

            max_len:
                BERT 输入序列的最大长度。
                每条输入都会被 padding / truncation 到 max_len。

            vocab:
                BERT 预训练模型对应的词表。
                注意这里通常不能自己重新构建词表，
                而应该使用预训练 BERT 自带的 vocab。
        """

        # 对 premise 和 hypothesis 分别转小写并分词，然后按样本重新配对
        #
        # dataset[:2] 取出:
        #   dataset[0] -> premises
        #   dataset[1] -> hypotheses
        #
        # 对每一组 sentences:
        #   [s.lower() for s in sentences] 先转小写
        #   tokenize(...) 再分词
        #
        # 最终 all_premise_hypothesis_tokens 的结构是:
        # [
        #     [premise_tokens_1, hypothesis_tokens_1],
        #     [premise_tokens_2, hypothesis_tokens_2],
        #     ...
        # ]
        all_premise_hypothesis_tokens = [
            [p_tokens, h_tokens]
            for p_tokens, h_tokens in zip(
                *[
                    tokenize([s.lower() for s in sentences])
                    for sentences in dataset[:2]
                ]
            )
        ]

        # SNLI 标签，shape: (样本数,)
        self.labels = torch.tensor(dataset[2])

        # 使用传入的 BERT 词表
        self.vocab = vocab

        # BERT 最大输入长度
        self.max_len = max_len
        
        # 对所有 premise-hypothesis 样本进行 BERT 输入预处理
        #
        # self.all_token_ids shape:
        #   (样本数, max_len)
        #
        # self.all_segments shape:
        #   (样本数, max_len)
        #
        # self.valid_lens shape:
        #   (样本数,)
        (self.all_token_ids,
         self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)

        print('read ' + str(len(self.all_token_ids)) + ' examples')


    def _preprocess(self, all_premise_hypothesis_tokens):
        """对所有 premise-hypothesis 样本进行预处理。

        主要工作:
            1. 截断过长的 premise 和 hypothesis；
            2. 添加 <cls> 和 <sep>；
            3. 构造 segment ids；
            4. 转换成 token ids；
            5. padding 到 max_len。
        """

        # 使用 4 个进程并行处理样本，加快预处理速度
        pool = multiprocessing.Pool(4)

        # pool.map 会对 all_premise_hypothesis_tokens 中的每个样本
        # 调用 self._mp_worker
        #
        # out 中每个元素是:
        #   (token_ids, segments, valid_len)
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)

        # 取出所有样本的 token_ids
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out
        ]

        # 取出所有样本的 segments
        all_segments = [
            segments for token_ids, segments, valid_len in out
        ]

        # 取出所有样本的 valid_len
        valid_lens = [
            valid_len for token_ids, segments, valid_len in out
        ]

        # 转成 tensor
        return (
            torch.tensor(all_token_ids, dtype=torch.long),
            torch.tensor(all_segments, dtype=torch.long),
            torch.tensor(valid_lens)
        )


    def _mp_worker(self, premise_hypothesis_tokens):
        """处理单条 premise-hypothesis 样本。

        输入:
            premise_hypothesis_tokens:
                [p_tokens, h_tokens]

                例如:
                    [
                        ['a', 'dog', 'is', 'running'],
                        ['an', 'animal', 'runs']
                    ]

        返回:
            token_ids:
                padding 后的 BERT 输入 token id，长度为 max_len。

            segments:
                padding 后的 segment id，长度为 max_len。

            valid_len:
                padding 前的有效长度。
        """

        # 拆出 premise tokens 和 hypothesis tokens
        p_tokens, h_tokens = premise_hypothesis_tokens

        # 如果 premise + hypothesis 太长，就进行截断
        self._truncate_pair_of_tokens(p_tokens, h_tokens)

        # 构造 BERT 输入 tokens 和 segments
        #
        # tokens:
        #   <cls> premise <sep> hypothesis <sep>
        #
        # segments:
        #   premise 部分为 0
        #   hypothesis 部分为 1
        tokens, segments = _get_tokens_and_segments(p_tokens, h_tokens)

        # 将 tokens 转成 token ids，并 padding 到 max_len
        #
        # self.vocab[tokens]:
        #   token 字符串列表 -> token id 列表
        #
        # [self.vocab['<pad>']] * (self.max_len - len(tokens)):
        #   不足 max_len 的部分补 <pad>
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (
            self.max_len - len(tokens)
        )

        # segments 也 padding 到 max_len
        # padding 部分补 0
        segments = segments + [0] * (
            self.max_len - len(segments)
        )

        # 有效长度是不包含 <pad> 的真实 token 数量
        valid_len = len(tokens)

        return token_ids, segments, valid_len


    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        """截断 premise 和 hypothesis，使二者加上特殊 token 后不超过 max_len。

        BERT 输入格式:
            <cls> premise <sep> hypothesis <sep>

        因此要预留 3 个特殊 token 位置:
            1 个 <cls>
            2 个 <sep>
        """

        # 如果 premise 和 hypothesis 总长度超过 max_len - 3，
        # 就不断删除较长句子的最后一个 token。
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        """根据索引 idx 返回一条样本。

        返回:
            ((token_ids, segments, valid_len), label)
        """

        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx]
        ), self.labels[idx]

    def __len__(self):
        """返回数据集中的样本数量。"""

        return len(self.all_token_ids)


class BERTClassifier(nn.Module):
    """基于预训练 BERT 的分类模型。

    用于 SNLI 自然语言推理任务。

    输入:
        inputs 是一个三元组:
            tokens_X:
                BERT 输入 token id。
                shape: (batch_size, max_len)

            segments_X:
                segment id。
                shape: (batch_size, max_len)
                0 表示前提句 premise 部分；
                1 表示假设句 hypothesis 部分。

            valid_lens_x:
                每条样本的有效长度，不包含 <pad>。
                shape: (batch_size,)

    输出:
        logits:
            三分类预测结果。
            shape: (batch_size, 3)
    """

    def __init__(self, bert, num_hiddens):
        """初始化 BERTClassifier。

        参数:
            bert:
                已经预训练好的 BERTModel。

            num_hiddens:
                BERT 模型的隐藏层维度。
        """
        super(BERTClassifier, self).__init__()

        # 复用预训练 BERT 的编码器部分
        #
        # encoder 的作用:
        #   输入 token_ids、segments、valid_lens
        #   输出每个 token 的上下文表示
        #
        # 输出 encoded_X shape:
        #   (batch_size, max_len, num_hiddens)
        self.encoder = bert.encoder
        
        # 复用预训练 BERT 中用于处理 <cls> 表示的 hidden 层
        #
        # 在前面 BERTModel 中通常是:
        #   nn.Sequential(
        #       nn.Linear(num_hiddens, num_hiddens),
        #       nn.Tanh()
        #   )
        #
        # 它的作用是把 <cls> 位置的表示再变换一下，
        # 得到适合句子级分类任务的向量。
        self.hidden = bert.hidden

        # 新建一个三分类输出层
        #
        # 如果使用的是 D2L 的 bert.base:
        #   num_hiddens = 768
        #   所以这里是 Linear(768, 3)
        #
        # 3 表示 SNLI 的三个类别:
        #   entailment, contradiction, neutral
        self.output = nn.Linear(num_hiddens, 3)

    def forward(self, inputs):
        """前向传播。

        参数:
            inputs:
                一个三元组:
                    tokens_X, segments_X, valid_lens_x

        返回:
            三分类 logits。
            shape: (batch_size, 3)
        """

        # 解包输入
        #
        # tokens_X shape:
        #   (batch_size, max_len)
        #
        # segments_X shape:
        #   (batch_size, max_len)
        #
        # valid_lens_x shape:
        #   (batch_size,)
        tokens_X, segments_X, valid_lens_x = inputs

        # 使用 BERT encoder 对输入句子对进行编码
        #
        # encoded_X shape:
        #   (batch_size, max_len, num_hiddens)
        #
        # 其中每个 token 都会得到一个上下文表示。
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)

        # 取 <cls> 位置的输出表示
        #
        # BERT 输入格式是:
        #   <cls> premise <sep> hypothesis <sep>
        #
        # 所以第 0 个位置 encoded_X[:, 0, :] 就是 <cls> 的表示。
        #
        # encoded_X[:, 0, :] shape:
        #   (batch_size, num_hiddens)
        cls_encoding = encoded_X[:, 0, :]

        # 经过 BERT 自带的 hidden 层
        #
        # hidden(cls_encoding) shape:
        #   (batch_size, num_hiddens)
        hidden_X = self.hidden(cls_encoding)

        # 经过三分类输出层
        #
        # output(hidden_X) shape:
        #   (batch_size, 3)
        #
        # 返回的是 logits，不是概率。
        return self.output(hidden_X)


def read_snli(data_dir, is_train):
    """读取 SNLI 数据集文件，并解析成 premises、hypotheses 和 labels。

    参数:
        data_dir:
            SNLI 数据集目录。

        is_train:
            True 表示读取训练集 snli_1.0_train.txt；
            False 表示读取测试集 snli_1.0_test.txt。

    返回:
        premises:
            前提句子列表。

        hypotheses:
            假设句子列表。

        labels:
            标签列表。
            entailment -> 0
            contradiction -> 1
            neutral -> 2
    """

    def extract_text(s):
        """清洗 SNLI 文件中的句子文本。

        D2L 代码中使用的是 SNLI 文件中的解析树字段，
        这些字段通常包含括号。
        例如:
            "( A dog ) ( is running )"

        这里会：
            1. 删除左括号；
            2. 删除右括号；
            3. 把多个连续空格替换成一个空格；
            4. 去掉首尾空白。
        """

        # 删除左括号
        s = re.sub('\\(', '', s)

        # 删除右括号
        s = re.sub('\\)', '', s)

        # 把两个或多个连续空白替换成一个空格
        s = re.sub('\\s{2,}', ' ', s)

        return s.strip()

    # 定义标签到数字编号的映射
    label_set = {
        'entailment': 0,
        'contradiction': 1,
        'neutral': 2
    }

    # 根据 is_train 选择读取训练集或测试集
    file_name = os.path.join(
        data_dir,
        'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt'
    )

    # 读取文件
    #
    # SNLI 文件是 tab 分隔格式
    # 第一行是表头，所以用 f.readlines()[1:] 跳过第一行
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]

    # 只保留标签在 label_set 中的样本
    # 有些样本标签可能是 '-'，表示无效或无一致标注，需要过滤掉
    #
    # row[0] 是标签
    # row[1] 是 premise 的解析树文本
    # row[2] 是 hypothesis 的解析树文本
    premises = [
        extract_text(row[1])
        for row in rows
        if row[0] in label_set
    ]

    hypotheses = [
        extract_text(row[2])
        for row in rows
        if row[0] in label_set
    ]

    labels = [
        label_set[row[0]]
        for row in rows
        if row[0] in label_set
    ]

    return premises, hypotheses, labels


def load_pretrained_model(model_dir, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    # 定义空词表以加载预定义词表
    vocab = Vocab()

    vocab.idx_to_token = json.load(
        open(
            os.path.join(model_dir, 'vocab.json')
        )
    )

    vocab.token_to_idx = {
        token: idx
        for idx, token in enumerate(
            vocab.idx_to_token
        )
    }

    bert = BERTModel(
        len(vocab),
        num_hiddens,
        norm_shape=[num_hiddens],
        ffn_num_input=num_hiddens,
        ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_len=max_len,
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        hid_in_features=num_hiddens,
        mlm_in_features=num_hiddens,
        nsp_in_features=num_hiddens
    )

    # 加载预训练BERT参数
    bert.load_state_dict(
        torch.load(
            os.path.join(model_dir, 'pretrained.params')
        )
    )

    return bert, vocab


def train(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=try_all_gpus()):
    """使用单 GPU / 多 GPU 训练模型。

    参数:
        net:
            要训练的模型。
            例如 BERTClassifier。

        train_iter:
            训练集 DataLoader。
            每次迭代返回一个 batch：
                features, labels

        test_iter:
            测试集 DataLoader。
            用于每个 epoch 结束后评估测试准确率。

        loss:
            损失函数。
            例如:
                nn.CrossEntropyLoss(reduction='none')

        trainer:
            优化器。
            例如:
                torch.optim.Adam(net.parameters(), lr=1e-4)

        num_epochs:
            训练轮数。
            1 个 epoch 表示完整遍历一遍 train_iter。

        devices:
            训练设备列表。
            例如:
                [torch.device('cuda:0')]
                [torch.device('cuda:0'), torch.device('cuda:1')]
                [torch.device('cpu')]
    """

    # timer 用于统计训练耗时
    # num_batches 表示一个 epoch 中有多少个 batch
    timer, num_batches = Timer(), len(train_iter)

    # 每个 epoch 大约打印 5 次
    # max(1, ...) 是为了防止 num_batches < 5 时 num_batches // 5 等于 0
    print_interval = max(1, num_batches // 5)

    # 使用 DataParallel 包装模型，实现多 GPU 并行训练
    #
    # device_ids=devices:
    #   指定参与训练的设备
    #
    # .to(devices[0]):
    #   将主模型放到第一个设备上
    #
    # 例如 devices = [cuda:0, cuda:1]，
    # 那么主设备是 cuda:0。
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # 外层循环：训练 num_epochs 轮
    for epoch in range(num_epochs):

        # 每个 epoch 重新初始化统计器
        #
        # metric[0]: 累计训练 loss 总和
        # metric[1]: 累计预测正确数量
        # metric[2]: 累计样本数量，用于计算平均 loss
        # metric[3]: 累计标签数量，用于计算准确率
        metric = Accumulator(4)

        # 内层循环：遍历训练集中的每个 batch
        for i, (features, labels) in enumerate(train_iter):

            # 开始计时
            timer.start()
            
            # 训练一个 batch
            #
            # l:
            #   当前 batch 的 loss 总和
            #
            # acc:
            #   当前 batch 中预测正确的样本数量
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices
            )

            # 累加当前 batch 的指标
            #
            # labels.shape[0]:
            #   当前 batch 的样本数
            #
            # labels.numel():
            #   当前 batch 的标签数量
            #   对分类任务来说，通常等于 labels.shape[0]
            metric.add(l, acc, labels.shape[0], labels.numel())

            # 停止计时
            timer.stop()

            # 训练过程中定期打印
            if (i + 1) % print_interval == 0 or i == num_batches - 1:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[3]

                print(
                    f'epoch {epoch + 1:2d}/{num_epochs}, '
                    f'batch {i + 1:4d}/{num_batches}, '
                    f'train loss {train_loss:.4f}, '
                    f'train acc {train_acc:.4f}'
                )

        # 每个 epoch 结束后，在测试集上评估准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)

        # 打印当前 epoch 结束后的测试准确率
        print(
            f'epoch {epoch + 1:2d}/{num_epochs} finished, '
            f'test acc {test_acc:.4f}'
        )

    # 所有 epoch 训练完成后，打印最终结果
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')

    # 打印训练速度
    #
    # metric[2]:
    #   最后一个 epoch 中处理的样本数
    #
    # metric[2] * num_epochs:
    #   估算整个训练过程中处理的样本数
    #
    # timer.sum():
    #   总训练耗时
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


def train_batch(net, X, y, loss, trainer, devices):
    """训练一个小批量 batch。

    参数:
        net:
            要训练的模型，例如 BERTClassifier。

        X:
            当前 batch 的输入特征。

            对普通模型来说，X 可能是一个 tensor。
            对 BERT 来说，X 通常是多个 tensor 组成的 list：
                [token_ids, segments, valid_lens]

        y:
            当前 batch 的真实标签。
            shape: (batch_size,)

        loss:
            损失函数。
            例如:
                nn.CrossEntropyLoss(reduction='none')

        trainer:
            优化器。
            例如:
                torch.optim.Adam(net.parameters(), lr=1e-4)

        devices:
            训练设备列表。
            例如:
                [torch.device('cuda:0')]
                [torch.device('cpu')]

    返回:
        train_loss_sum:
            当前 batch 的 loss 总和。

        train_acc_sum:
            当前 batch 中预测正确的样本数量。
    """

    # 如果 X 是 list，说明输入由多个张量组成。
    #
    # BERT 微调时，X 通常是：
    # [
    #     token_ids,    shape: (batch_size, max_len)
    #     segments,     shape: (batch_size, max_len)
    #     valid_lens    shape: (batch_size,)
    # ]
    #
    # 需要把每个张量都移动到主设备 devices[0] 上。
    if isinstance(X, (list, tuple)):
        X = [x.to(devices[0]) for x in X]
    
    # 如果 X 是普通 Tensor，直接移动到主设备 devices[0] 上。
    else:
        X = X.to(devices[0])

    # 将真实标签 y 也移动到同一个设备上。
    #
    # 注意：
    # 模型和数据必须在同一个设备上，否则会报 device mismatch 错误。
    y = y.to(devices[0])
    
    # 将模型切换到训练模式。
    #
    # 训练模式下：
    # 1. Dropout 会生效；
    # 2. BatchNorm 会使用当前 batch 的统计量。
    #
    # 对 BERT 来说，主要影响 Dropout。
    net.train()

    # 梯度清零。
    #
    # PyTorch 默认会累积梯度，
    # 所以每个 batch 反向传播前都要清空上一轮的梯度。
    trainer.zero_grad()

    # 前向传播。
    #
    # 对 SNLI 三分类任务来说：
    # pred shape: (batch_size, 3)
    #
    # 其中 3 表示三个类别：
    # 0: entailment
    # 1: contradiction
    # 2: neutral
    pred = net(X)

    # 计算当前 batch 的损失。
    #
    # 如果 loss = nn.CrossEntropyLoss(reduction='none')，
    # 那么 l 的 shape 是：
    # (batch_size,)
    #
    # 也就是每个样本都有一个单独的 loss。
    l = loss(pred, y)

    # 对当前 batch 中所有样本的 loss 求和，
    # 得到一个标量，然后进行反向传播。
    #
    # backward() 要求输出通常是标量，
    # 所以这里使用 l.sum()。
    l.sum().backward()

    # 根据梯度更新模型参数。
    trainer.step()

    # 当前 batch 的 loss 总和。
    #
    # 注意：
    # 这里返回的是 sum，不是 mean。
    # 后面通常会用：
    # metric[0] / metric[2]
    # 计算平均 loss。
    #
    # 这里推荐转成 Python 数值，原因是：
    # detach(): 从计算图中分离
    # cpu(): 移动到 CPU
    # item(): 转成 Python float
    # 这样不会把计算图意外保存在统计器里。
    train_loss_sum = l.sum().detach().cpu().item()

    # 当前 batch 中预测正确的样本数量。
    #
    # accuracy(pred, y) 返回的不是准确率比例，
    # 而是当前 batch 中预测正确的个数。
    train_acc_sum = accuracy(pred, y)

    return train_loss_sum, train_acc_sum


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在 GPU 或指定设备上计算模型在数据集上的准确率。

    参数:
        net:
            要评估的模型，例如 BERTClassifier。

        data_iter:
            数据迭代器，例如 test_iter。
            每次迭代返回一个 batch：
                X, y

            对于普通模型：
                X 可能是一个 tensor。

            对于 BERT 微调：
                X 通常是一个列表或元组，里面包含：
                    token_ids,
                    segments,
                    valid_lens

        device:
            指定评估设备。
            如果为 None，则自动从模型参数所在设备推断。

    返回:
        accuracy:
            整个数据集上的准确率。
    """

    # 如果 net 是 PyTorch 的 nn.Module 模型
    if isinstance(net, nn.Module):
        
        # 切换到评估模式
        #
        # 这一步非常重要：
        # 1. Dropout 会关闭
        # 2. BatchNorm 会使用固定统计量
        #
        # 对 BERT 来说，主要影响 Dropout。
        net.eval()
        
        # 如果没有手动指定 device，
        # 就从模型第一个参数所在的位置推断 device。
        #
        # 例如：
        # 如果模型在 cuda:0 上，则 device = cuda:0
        # 如果模型在 cpu 上，则 device = cpu
        if not device:
            device = next(iter(net.parameters())).device

    # 创建累加器，用于累计两个数：
    #
    # metric[0]：预测正确的样本数量
    # metric[1]：样本总数量
    metric = Accumulator(2)

    # 评估阶段不需要计算梯度
    #
    # torch.no_grad() 可以：
    # 1. 减少显存占用
    # 2. 加快推理速度
    # 3. 防止无意义地构建计算图
    with torch.no_grad():

        # 遍历整个数据集
        for X, y in data_iter:

            # 如果 X 是 list，说明输入由多个张量组成
            #
            # 例如 BERT 微调时：
            # X = [
            #     token_ids,
            #     segments,
            #     valid_lens
            # ]
            #
            # 这些张量都需要移动到同一个 device。
            if isinstance(X, list):
                X = [x.to(device) for x in X]

            # 如果 X 是普通 tensor，
            # 直接移动到 device。
            else:
                X = X.to(device)

            # 标签 y 也要移动到同一个 device。
            y = y.to(device)

            # net(X) 得到模型预测结果
            #
            # accuracy(net(X), y) 返回当前 batch 中预测正确的样本数
            #
            # y.numel() 返回当前 batch 的样本数
            #
            # metric.add(...) 将二者累加起来
            metric.add(accuracy(net(X), y), y.numel())

    # 整个数据集准确率 =
    # 总预测正确数 / 总样本数
    return metric[0] / metric[1]


def accuracy(y_hat, y):
    """计算预测正确的样本数量。

    参数:
        y_hat:
            模型输出的预测结果。

            对于多分类任务，通常是 logits，形状为：
                (batch_size, num_classes)

            例如 SNLI 三分类任务中：
                y_hat.shape = (batch_size, 3)

        y:
            真实标签，形状为：
                (batch_size,)

            例如：
                tensor([0, 2, 1, 0])

    返回:
        当前 batch 中预测正确的样本数量，类型是 float。
    """

    # 如果 y_hat 是二维张量，并且第 1 维大于 1，
    # 说明 y_hat 是多分类输出。
    #
    # 例如：
    # y_hat.shape = (4, 3)
    #
    # 每一行对应一个样本，每一列对应一个类别的预测分数。
    # 此时需要取每一行最大值所在的位置，作为模型预测类别。
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    
    # 将预测结果 y_hat 转换成和真实标签 y 相同的数据类型。
    #
    # 然后逐元素比较：
    #   预测类别 == 真实类别
    #
    # cmp 是一个 bool 类型张量。
    #
    # 例如：
    # y_hat = tensor([0, 1, 2, 1])
    # y     = tensor([0, 2, 2, 1])
    #
    # cmp = tensor([True, False, True, True])
    cmp = y_hat.type(y.dtype) == y

    # 将 bool 张量转换成和 y 相同的数据类型。
    #
    # True  -> 1
    # False -> 0
    #
    # 然后 sum() 求和，得到预测正确的样本数量。
    #
    # 最后用 float(...) 转成 Python 浮点数返回。
    return float(cmp.type(y.dtype).sum())


if __name__ == "__main__":

    # =========================
    # 1. 选择训练设备
    # =========================

    # try_all_gpus() 会返回所有可用 GPU。
    # 如果没有 GPU，则返回 [torch.device('cpu')]。
    #
    # 例如：
    #   有 GPU: [device(type='cuda', index=0)]
    #   没有 GPU: [device(type='cpu')]
    devices = try_all_gpus()

    # =========================
    # 2. 加载 D2L 提供的预训练 BERT-small
    # =========================

    # load_pretrained_model 会加载：
    #   1. 预训练 BERT 模型结构；
    #   2. 预训练参数 pretrained.params；
    #   3. 预训练词表 vocab.json。
    #
    # "../model/bert.base.torch" 是本地预训练模型目录。
    #
    # bert.base 的结构参数通常是：
    #   num_hiddens = 768
    #   ffn_num_hiddens = 3072
    #   num_heads = 14
    #   num_layers = 12
    #   max_len = 512
    #
    # 返回：
    #   bert:
    #       预训练好的 BERTModel。
    #
    #   vocab:
    #       预训练 BERT 对应的词表。
    #       后面处理 SNLI 数据时必须使用这个 vocab，
    #       不能重新用 SNLI 训练集构建词表。
    bert, vocab = load_pretrained_model(
        "../model/bert.base.torch",
        num_hiddens=768,
        ffn_num_hiddens=3072,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        max_len=512,
        devices=devices
    )

    # =========================
    # 3. 设置 SNLI 数据参数
    # =========================

    # batch_size:
    #   每个 batch 中有 32 条 SNLI 样本。
    #
    # max_len:
    #   每条 BERT 输入序列最大长度为 128。
    #   注意：虽然预训练 BERT-small 支持 max_len=512，
    #   但微调时实际输入可以只用 128。
    #
    # num_workers:
    #   DataLoader 使用 8 个子进程加载数据。
    batch_size, max_len, num_workers = 32, 128, 8

    # SNLI 数据集所在目录
    # 目录下通常包含：
    #   snli_1.0_train.txt
    #   snli_1.0_test.txt
    data_dir = "../data/snli_1.0"

    # =========================
    # 4. 构建 SNLI 的 BERT 数据集
    # =========================

    # read_snli(data_dir, True):
    #   读取训练集，返回：
    #       premises,
    #       hypotheses,
    #       labels
    #
    # SNLIBERTDataset 会把 premise 和 hypothesis 拼成 BERT 输入：
    #
    #   <cls> premise <sep> hypothesis <sep>
    #
    # 并生成：
    #   token_ids,
    #   segments,
    #   valid_lens,
    #   labels
    train_set = SNLIBERTDataset(
        read_snli(data_dir, True),
        max_len,
        vocab
    )

    # 构建测试集。
    # 注意：测试集也必须使用同一个预训练 BERT 词表 vocab。
    test_set = SNLIBERTDataset(
        read_snli(data_dir, False),
        max_len, 
        vocab
    )

    # =========================
    # 5. 构建 DataLoader
    # =========================

    # 训练集 DataLoader。
    #
    # shuffle=True:
    #   每个 epoch 打乱训练样本顺序。
    #
    # 每次迭代返回：
    #   X, y
    #
    # 其中：
    #   X = (token_ids, segments, valid_lens)
    #   y = labels
    train_iter = torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 测试集 DataLoader。
    #
    # 测试集一般不需要 shuffle。
    test_iter = torch.utils.data.DataLoader(
        test_set,
        batch_size,
        num_workers=num_workers
    )

    # =========================
    # 6. 构建 BERT 分类模型
    # =========================

    # BERTClassifier 会复用：
    #   bert.encoder
    #   bert.hidden
    #
    # 然后新增一个三分类输出层：
    #   nn.Linear(768, 3)
    #
    # 3 对应 SNLI 三个类别：
    #   entailment
    #   contradiction
    #   neutral
    net = BERTClassifier(bert, num_hiddens=768)

    # =========================
    # 7. 定义损失函数
    # =========================

    # reduction='none' 表示：
    #   不直接对 batch loss 求平均，
    #   而是返回每个样本各自的 loss。
    #
    # 这样在 train_batch 中可以执行：
    #   l.sum().backward()
    #
    # 同时也方便统计平均 loss：
    #   累计 loss 总和 / 累计样本数
    loss = nn.CrossEntropyLoss(reduction='none')

    # =========================
    # 8. 定义优化器
    # =========================

    # 学习率
    lr = 1e-5

    # 训练 epoch 数
    num_epochs = 5

    # Adam 优化器。
    #
    # net.parameters() 包含：
    #   1. BERT encoder 参数；
    #   2. BERT hidden 层参数；
    #   3. 新增分类层 output 参数。
    #
    # 因此微调时不仅训练分类层，也会微调 BERT 本身。
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    # =========================
    # 9. 开始微调训练
    # =========================

    # train 会执行：
    #   多 epoch 训练；
    #   每个 batch 前向传播、计算 loss、反向传播、更新参数；
    #   每个 epoch 后在 test_iter 上评估准确率。
    train(
        net,
        train_iter,
        test_iter,
        loss,
        trainer,
        num_epochs,
        devices
    )
