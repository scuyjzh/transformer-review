import os
import random
import nltk
import torch

from transformer import Vocab


def _read_wiki(data_dir):
    """读取 WikiText-2 训练集，并将文本处理成段落列表。

    参数:
        data_dir: WikiText-2 数据集所在目录。
            目录中应该包含文件 wiki.train.tokens。

    返回:
        paragraphs: 段落列表。
            每个元素是一个段落；
            每个段落又是由多个句子组成的列表。

            例如:
                [
                    ['this is sentence one', 'this is sentence two'],
                    ['another paragraph sentence one', 'another paragraph sentence two']
                ]

    处理流程:
        1. 读取 wiki.train.tokens 文件；
        2. 将每一行文本转成小写；
        3. 使用 NLTK 切分句子；
        4. 只保留至少包含两个句子的段落；
        5. 打乱段落顺序。
    """

    # 拼接 WikiText-2 训练文件路径
    # 例如 data_dir = '/root/data/wikitext-2'
    # file_name = '/root/data/wikitext-2/wiki.train.tokens'
    file_name = os.path.join(data_dir, 'wiki.train.tokens')

    # 读取训练文件中的所有行
    # 每一行通常可以看作一个段落
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # 确保 NLTK 的 Punkt 句子切分器已经下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # 如果环境提示缺少 punkt_tab，可以保留这段兼容处理
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    paragraphs = []

    for line in lines:
        # 去掉首尾空白，并转换成小写
        line = line.strip().lower()

        # 跳过空行
        if not line:
            continue

        # 使用 NLTK 进行句子切分
        # 例如:
        # "this is great ! why not ?"
        # 可能被切成:
        # ["this is great !", "why not ?"]
        sentences = nltk.tokenize.sent_tokenize(line)

        # 去掉每个句子首尾空白，并过滤空句子
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # 只保留至少包含两个句子的段落
        # 因为 BERT 的 NSP 任务需要 sentence A 和 sentence B
        if len(sentences) >= 2:
            paragraphs.append(sentences)

    # 打乱段落顺序，避免训练时总是按照原始文本顺序读取
    random.shuffle(paragraphs)

    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """构造 BERT 下一句预测任务 NSP 的一个训练样本。

    参数:
        sentence:
            当前句子，也就是句子 A。
            例如:
                'i like deep learning'

        next_sentence:
            当前句子在原文中的真实下一句，也就是原始句子 B。
            例如:
                'bert is a pretraining model'

        paragraphs:
            全部段落数据。
            一般是一个嵌套列表：
                [
                    ['sentence1', 'sentence2', 'sentence3'],
                    ['sentence4', 'sentence5'],
                    ...
                ]
            每个 paragraph 是一个句子列表。

    返回:
        sentence:
            句子 A，保持不变。

        next_sentence:
            句子 B。
            如果构造正样本，则是真实下一句；
            如果构造负样本，则是从其他段落中随机抽取的一句话。

        is_next:
            NSP 标签。
            True 表示 next_sentence 是 sentence 的真实下一句；
            False 表示 next_sentence 是随机抽取的句子，不是真实下一句。
    """

    # 以 50% 的概率构造正样本
    # random.random() 会生成 [0, 1) 之间的随机小数
    if random.random() < 0.5:
        # 保留原来的真实下一句
        # sentence 和 next_sentence 在原文中是连续的
        is_next = True

    else:
        # 以 50% 的概率构造负样本
        # 从所有 paragraphs 中随机选一个段落，
        # 再从这个段落中随机选一句话，
        # 用它替换原来的真实下一句
        next_sentence = random.choice(random.choice(paragraphs))

        # 此时 next_sentence 通常不是 sentence 的真实下一句
        is_next = False

    # 返回句子 A、句子 B、以及 NSP 标签
    return sentence, next_sentence, is_next


def _get_tokens_and_segments(tokens_a, tokens_b=None):
    """构造 BERT 输入所需的 token 序列和 segment 序列。

    BERT 的输入通常需要在句子开头添加 <cls>，
    在句子结尾添加 <sep>。

    如果只有一个句子，则形式为:
        <cls> tokens_a <sep>

    如果有两个句子，则形式为:
        <cls> tokens_a <sep> tokens_b <sep>

    参数:
        tokens_a: 第一个句子的 token 列表。
            例如:
                ['i', 'love', 'deep', 'learning']

        tokens_b: 第二个句子的 token 列表，默认为 None。
            如果不为 None，表示输入是句子对。
            例如:
                ['it', 'is', 'interesting']

    返回:
        tokens: 加入 <cls> 和 <sep> 后的 token 序列。
        segments: 与 tokens 等长的片段索引列表。
            0 表示 token 属于句子 A；
            1 表示 token 属于句子 B。
    """

    # 构造句子 A 部分:
    # 在 tokens_a 前加 <cls>，在 tokens_a 后加 <sep>
    # 形式为: <cls> tokens_a <sep>
    tokens = ['<cls>'] + tokens_a + ['<sep>']

    # 为句子 A 部分构造 segment id
    # <cls>、tokens_a 和第一个 <sep> 都属于片段 A，所以 segment id 都是 0
    # 长度为 len(tokens_a) + 2，其中 2 表示 <cls> 和 <sep>
    segments = [0] * (len(tokens_a) + 2)
    
    # 如果存在句子 B，则继续拼接句子 B 部分
    if tokens_b is not None:
        # 在 tokens 后追加 tokens_b 和第二个 <sep>
        # 形式变为: <cls> tokens_a <sep> tokens_b <sep>
        tokens += tokens_b + ['<sep>']

        # 为句子 B 部分构造 segment id
        # tokens_b 和最后一个 <sep> 都属于片段 B，所以 segment id 都是 1
        # 长度为 len(tokens_b) + 1，其中 1 表示最后的 <sep>
        segments += [1] * (len(tokens_b) + 1)

    # 返回 BERT 输入 token 序列及其对应的 segment 序列
    return tokens, segments


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """从一个段落中构造 BERT 下一句预测 NSP 任务的数据。

    参数:
        paragraph:
            当前段落。
            通常是一个句子列表，并且每个句子已经被分词。
            例如:
                [
                    ['i', 'like', 'deep', 'learning'],
                    ['bert', 'is', 'a', 'pretraining', 'model'],
                    ['it', 'uses', 'mlm', 'and', 'nsp']
                ]

        paragraphs:
            全部段落数据。
            用于在构造负样本时随机抽取一句话。
            例如:
                [
                    [
                        ['i', 'like', 'deep', 'learning'],
                        ['bert', 'is', 'a', 'pretraining', 'model']
                    ],
                    [
                        ['the', 'weather', 'is', 'cold'],
                        ['i', 'stay', 'at', 'home']
                    ]
                ]

        vocab:
            词表对象。
            当前函数中没有直接使用 vocab，
            但后续构造 MLM 数据、padding、token 转 id 时会用到。
            这里保留该参数通常是为了和整体数据处理流程保持一致。

        max_len:
            BERT 输入序列的最大长度。
            由于 BERT 输入格式为:
                <cls> tokens_a <sep> tokens_b <sep>
            所以需要额外预留 3 个特殊 token 的位置。

    返回:
        nsp_data_from_paragraph:
            当前段落中构造出的 NSP 样本列表。
            每个元素是:
                (tokens, segments, is_next)

            其中:
                tokens:
                    加入 <cls> 和 <sep> 后的 token 序列。

                segments:
                    与 tokens 等长的 segment id。
                    0 表示句子 A，1 表示句子 B。

                is_next:
                    NSP 标签。
                    True 表示句子 B 是句子 A 的真实下一句；
                    False 表示句子 B 是随机抽取的句子。
    """

    # 用于保存从当前 paragraph 构造出来的所有 NSP 样本
    nsp_data_from_paragraph = []

    # 遍历当前段落中的相邻句子对
    # 如果 paragraph 有 n 个句子，则可以构造 n - 1 个相邻句子对：
    # paragraph[0] 和 paragraph[1]
    # paragraph[1] 和 paragraph[2]
    # ...
    # paragraph[n-2] 和 paragraph[n-1]
    for i in range(len(paragraph) - 1):
        
        # 构造一个 NSP 句子对样本
        #
        # paragraph[i]:
        #   当前句子，也就是句子 A
        #
        # paragraph[i + 1]:
        #   当前句子的真实下一句
        #
        # _get_next_sentence 会以 50% 概率保留真实下一句，
        # 以 50% 概率随机替换成其他句子。
        #
        # 返回:
        #   tokens_a: 句子 A
        #   tokens_b: 句子 B，可能是真实下一句，也可能是随机句子
        #   is_next: True / False，表示 tokens_b 是否是真实下一句
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs
        )

        # 判断加入特殊 token 后是否超过最大长度 max_len
        #
        # BERT 输入格式为:
        #   <cls> tokens_a <sep> tokens_b <sep>
        #
        # 所以总长度为:
        #   len(tokens_a) + len(tokens_b) + 3
        #
        # 其中 3 表示:
        #   1 个 <cls>
        #   2 个 <sep>
        #
        # 如果总长度超过 max_len，则跳过该样本
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue

        # 构造 BERT 输入 token 序列和 segment 序列
        #
        # tokens:
        #   ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
        #
        # segments:
        #   句子 A 部分为 0
        #   句子 B 部分为 1
        tokens, segments = _get_tokens_and_segments(tokens_a, tokens_b)

        # 保存当前 NSP 样本
        # is_next 是 NSP 标签
        nsp_data_from_paragraph.append((tokens, segments, is_next))

    # 返回当前段落构造出的所有 NSP 样本
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """为 BERT 的 MLM 任务构造带 mask 的输入 token 序列。

    参数:
        tokens:
            原始 token 序列。
            通常已经包含 <cls> 和 <sep>。
            例如:
                ['<cls>', 'i', 'like', 'deep', 'learning', '<sep>']

        candidate_pred_positions:
            可以被选中做 MLM 预测的位置列表。
            通常不包含 <cls> 和 <sep> 的位置。
            例如:
                [1, 2, 3, 4]

        num_mlm_preds:
            本条样本中需要预测的 token 数量。
            通常约等于 token 总数的 15%。

        vocab:
            词表对象。
            用于随机选择一个 token 来替换原 token。

    返回:
        mlm_input_tokens:
            替换后的 token 序列。
            某些位置可能被替换成 <mask>、随机 token，或者保持不变。

        pred_positions_and_labels:
            被选中预测的位置及其真实标签。
            每个元素形如:
                (被预测的位置, 原始 token)
            例如:
                [(3, 'deep'), (4, 'learning')]
    """

    # 复制一份 tokens，作为 MLM 输入序列
    # 注意不能直接修改原始 tokens，否则后面无法拿到真实标签
    mlm_input_tokens = [token for token in tokens]

    # 保存被预测位置和真实标签
    # 例如:
    # [(3, 'deep'), (4, 'learning')]
    pred_positions_and_labels = []

    # 打乱候选预测位置
    # 这样可以随机选择一部分 token 作为 MLM 预测目标
    random.shuffle(candidate_pred_positions)

    # 遍历候选位置，逐个决定是否选为 MLM 预测位置
    for mlm_pred_position in candidate_pred_positions:
        
        # 如果已经选够了 num_mlm_preds 个预测位置，就停止
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break

        # masked_token 表示当前位置最终要替换成什么
        masked_token = None

        # 80% 的概率：把当前位置 token 替换成 <mask>
        if random.random() < 0.8:
            masked_token = '<mask>'

        else:
            # 剩下 20% 的情况中：
            # 其中一半，也就是总概率 10%，保持原 token 不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]

            # 另一半，也就是总概率 10%，随机替换成词表中的某个 token
            else:
                masked_token = random.choice(vocab.idx_to_token)

        # 在 MLM 输入序列中替换当前位置的 token
        mlm_input_tokens[mlm_pred_position] = masked_token

        # 保存当前位置和原始真实 token
        # 注意：标签永远是原始 token，而不是替换后的 token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position])
        )

    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """根据输入 tokens 构造 BERT 掩蔽语言模型 MLM 任务的数据。

    参数:
        tokens:
            原始 token 序列，通常已经包含 <cls> 和 <sep>。
            例如:
                ['<cls>', 'i', 'like', 'deep', 'learning', '<sep>']

        vocab:
            词表对象。
            用于：
                1. 将 token 转换成 token id；
                2. 从词表中随机抽 token；
                3. 将 MLM 标签 token 转换成 label id。

    返回:
        mlm_input_token_ids:
            被处理后的输入 token id 序列。
            某些位置可能被替换成 <mask>、随机 token，或者保持不变。
            shape 可以理解为: (len(tokens),)

        pred_positions:
            被选中做 MLM 预测的位置索引列表。
            例如:
                [2, 4]

        mlm_pred_label_ids:
            被预测位置对应的真实 token id 标签。
            例如:
                vocab[['like', 'learning']]
    """

    # 保存可以被选中作为 MLM 预测目标的位置
    candidate_pred_positions = []

    # 遍历 tokens 中的每个 token 及其位置 i
    for i, token in enumerate(tokens):

        # BERT 的 MLM 任务不会预测特殊 token
        # 也就是 <cls> 和 <sep> 不会被 mask，也不会作为预测目标
        if token in ['<cls>', '<sep>']:
            continue

        # 普通 token 的位置可以作为候选预测位置
        candidate_pred_positions.append(i)

    # MLM 中通常随机选择 15% 的 token 作为预测目标
    #
    # round(len(tokens) * 0.15):
    #   计算大约 15% 的位置数量
    #
    # max(1, ...):
    #   保证至少选择 1 个 token 进行预测
    #
    # 注意这里用的是 len(tokens)，包含 <cls> 和 <sep>，
    # 这是 D2L 教学代码中的简化写法。
    # 更推荐写成:
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))
    # num_mlm_preds = max(1, round(len(tokens) * 0.15))

    # 根据候选位置随机选择 num_mlm_preds 个位置，
    # 并按照 BERT 的 80/10/10 规则替换输入 token：
    #   80% 替换成 <mask>
    #   10% 保持原 token 不变
    #   10% 替换成随机 token
    #
    # mlm_input_tokens:
    #   替换后的 token 序列
    #
    # pred_positions_and_labels:
    #   被预测位置及其原始真实 token
    #   例如:
    #       [(4, 'learning'), (2, 'like')]
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )

    # 按照位置索引从小到大排序
    # 这样 pred_positions 和 mlm_pred_labels 的顺序更稳定
    #
    # 例如原来:
    #   [(4, 'learning'), (2, 'like')]
    #
    # 排序后:
    #   [(2, 'like'), (4, 'learning')]
    pred_positions_and_labels = sorted(
        pred_positions_and_labels,
        key=lambda x: x[0]
    )

    # 取出所有被预测的位置
    #
    # 例如:
    #   [(2, 'like'), (4, 'learning')]
    # 得到:
    #   [2, 4]
    pred_positions = [v[0] for v in pred_positions_and_labels]

    # 取出所有被预测位置对应的真实 token 标签
    #
    # 例如:
    #   [(2, 'like'), (4, 'learning')]
    # 得到:
    #   ['like', 'learning']
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    # 将替换后的输入 tokens 转换成 token ids
    # 将真实标签 tokens 转换成 label ids
    #
    # vocab[mlm_input_tokens]:
    #   ['<cls>', 'i', '<mask>', 'deep', 'learning', '<sep>']
    #   -> [2, 10, 4, 36, 48, 3]
    #
    # vocab[mlm_pred_labels]:
    #   ['like', 'learning']
    #   -> [25, 48]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """对 BERT 预训练样本进行 padding，整理成模型训练需要的张量列表。

    参数:
        examples:
            BERT 预训练样本列表。
            每个元素是一个五元组:
                (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)

            其中:
                token_ids:
                    MLM 替换后的输入 token id 序列。

                pred_positions:
                    MLM 中需要预测的位置索引。

                mlm_pred_label_ids:
                    MLM 中被预测位置对应的真实 token id 标签。

                segments:
                    segment id 序列，0 表示句子 A，1 表示句子 B。

                is_next:
                    NSP 标签。
                    True 表示句子 B 是句子 A 的真实下一句；
                    False 表示句子 B 不是句子 A 的真实下一句。

        max_len:
            BERT 输入序列的最大长度。
            token_ids 和 segments 都会被 padding 到 max_len。

        vocab:
            词表对象，用于获取 <pad> 的 token id。

    返回:
        all_token_ids:
            padding 后的 token id 张量列表。
            每个张量 shape: (max_len,)

        all_segments:
            padding 后的 segment id 张量列表。
            每个张量 shape: (max_len,)

        valid_lens:
            每条样本的有效长度，不包括 <pad>。
            每个张量是标量。

        all_pred_positions:
            padding 后的 MLM 预测位置张量列表。
            每个张量 shape: (max_num_mlm_preds,)

        all_mlm_weights:
            MLM loss 权重。
            真实预测位置为 1，padding 出来的预测位置为 0。
            每个张量 shape: (max_num_mlm_preds,)

        all_mlm_labels:
            padding 后的 MLM 标签张量列表。
            每个张量 shape: (max_num_mlm_preds,)

        nsp_labels:
            NSP 标签张量列表。
            每个张量是标量。
    """

    # 每条样本最多预测多少个 MLM token
    # BERT MLM 任务通常预测约 15% 的 token
    # 例如 max_len = 64，则 max_num_mlm_preds = round(64 * 0.15) = 10
    max_num_mlm_preds = round(max_len * 0.15)

    # 保存 padding 后的 token id
    all_token_ids = []

    # 保存 padding 后的 segment id
    all_segments = []

    # 保存每条样本的有效长度，也就是不包括 <pad> 的 token 数
    valid_lens = []

    # 保存 padding 后的 MLM 预测位置
    all_pred_positions = []

    # 保存 MLM loss 权重
    # 真实预测位置权重为 1，padding 出来的位置权重为 0
    all_mlm_weights = []

    # 保存 padding 后的 MLM 标签
    all_mlm_labels = []

    # 保存 NSP 标签
    nsp_labels = []

    # 遍历每一条样本
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:

        # 1. 对 token_ids 做 padding
        #
        # 原始 token_ids 长度可能小于 max_len
        # 用 <pad> 的 id 补齐到 max_len
        #
        # shape: (max_len,)
        all_token_ids.append(
            torch.tensor(
                token_ids + [vocab['<pad>']] * (max_len - len(token_ids)),
                dtype=torch.long
            )
        )

        # 2. 对 segments 做 padding
        #
        # segment padding 部分补 0
        # 因为 <pad> 不属于真实句子，补 0 即可
        #
        # shape: (max_len,)
        all_segments.append(
            torch.tensor(
                segments + [0] * (max_len - len(segments)),
                dtype=torch.long
            )
        )

        # 3. 保存有效长度
        #
        # valid_len 表示当前样本中非 <pad> 的 token 数量
        # 后续在 self-attention 中用来屏蔽 <pad>
        valid_lens.append(
            torch.tensor(
                len(token_ids),
                dtype=torch.float32
            )
        )

        # 4. 对 MLM 预测位置 pred_positions 做 padding
        #
        # 不同样本的 MLM 预测位置数量可能不同，
        # 所以要统一 padding 到 max_num_mlm_preds
        #
        # padding 的位置用 0 补齐
        # 注意：这些补出来的位置后面会通过 all_mlm_weights = 0 屏蔽掉
        #
        # shape: (max_num_mlm_preds,)
        all_pred_positions.append(
            torch.tensor(
                pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.long
            )
        )

        # 5. 构造 MLM loss 权重
        #
        # 真实 MLM 预测位置对应权重为 1.0
        # padding 出来的预测位置对应权重为 0.0
        #
        # 这样计算 MLM loss 时，padding 出来的预测位置不会影响损失
        #
        # shape: (max_num_mlm_preds,)
        all_mlm_weights.append(
            torch.tensor(
                [1.0] * len(mlm_pred_label_ids) +
                [0.0] * (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32
            )
        )

        # 6. 对 MLM 标签做 padding
        #
        # mlm_pred_label_ids 是被预测位置的真实 token id
        # 不足 max_num_mlm_preds 的地方用 0 补齐
        #
        # 注意：补齐的 0 不会真正参与 loss，
        # 因为对应位置的 all_mlm_weights 是 0
        #
        # shape: (max_num_mlm_preds,)
        all_mlm_labels.append(
            torch.tensor(
                mlm_pred_label_ids +
                [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                dtype=torch.long
            )
        )

        # 7. 保存 NSP 标签
        #
        # is_next 是 bool:
        # True  -> 1
        # False -> 0
        #
        # dtype=torch.long 是因为后面 CrossEntropyLoss 需要类别标签是 long 类型
        nsp_labels.append(
            torch.tensor(
                is_next,
                dtype=torch.long
            )
        )

    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


class _WikiTextDataset(torch.utils.data.Dataset):
    """WikiText-2 数据集类，用于构造 BERT 预训练数据。

    该类会把原始段落数据处理成 BERT 训练需要的格式，包括：
        1. 分词；
        2. 构建词表；
        3. 构造 NSP 下一句预测任务数据；
        4. 构造 MLM 掩蔽语言模型任务数据；
        5. padding 成固定长度；
        6. 支持通过索引读取单条样本。

    每条样本最终包含：
        token_ids:
            BERT 输入 token id，shape: (max_len,)

        segments:
            segment id，区分句子 A 和句子 B，shape: (max_len,)

        valid_len:
            有效长度，不包括 <pad>，shape: 标量

        pred_positions:
            MLM 需要预测的位置，shape: (max_num_mlm_preds,)

        mlm_weights:
            MLM loss 权重，真实预测位置为 1，padding 位置为 0，
            shape: (max_num_mlm_preds,)

        mlm_labels:
            MLM 真实标签，shape: (max_num_mlm_preds,)

        nsp_label:
            NSP 标签，True/False 转成 1/0。
    """

    def __init__(self, paragraphs, max_len):
        """初始化 WikiText 数据集。

        参数:
            paragraphs:
                原始段落数据。
                输入时 paragraphs[i] 是一个段落，
                每个段落是由多个句子字符串组成的列表。

                例如:
                    paragraphs = [
                        [
                            'i like deep learning',
                            'bert is a pretraining model'
                        ],
                        [
                            'the weather is cold',
                            'i stay at home'
                        ]
                    ]

            max_len:
                BERT 输入序列的最大长度。
                后续 token_ids 和 segments 都会被 padding 到 max_len。
        """

        # 1. 对每个段落中的每个句子进行分词
        #
        # 输入 paragraphs[i]:
        #   ['i like deep learning', 'bert is a pretraining model']
        #
        # tokenize(paragraph, token='word') 后:
        #   [
        #       ['i', 'like', 'deep', 'learning'],
        #       ['bert', 'is', 'a', 'pretraining', 'model']
        #   ]
        #
        # 所以处理后:
        #   paragraphs 是三重列表:
        #   paragraphs -> paragraph -> sentence -> token
        paragraphs = [
            tokenize(paragraph, token='word')
            for paragraph in paragraphs
        ]

        # 2. 把所有段落中的所有句子拉平成一个句子列表
        #
        # 原结构:
        #   paragraphs = [
        #       [sentence1, sentence2],
        #       [sentence3, sentence4]
        #   ]
        #
        # 变成:
        #   sentences = [sentence1, sentence2, sentence3, sentence4]
        #
        # 这里的每个 sentence 都是 token 列表。
        sentences = [
            sentence
            for paragraph in paragraphs
            for sentence in paragraph
        ]

        # 3. 根据所有句子构建词表
        #
        # min_freq=5:
        #   只保留出现次数至少为 5 的 token，
        #   低频 token 会被映射为 <unk>。
        #
        # reserved_tokens:
        #   预留 BERT 需要的特殊 token。
        #
        # 注意:
        #   Vocab 默认通常会额外加入 <unk>。
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>']
        )

        # 4. 构造 NSP 下一句预测任务数据
        #
        # examples 先保存 NSP 样本:
        #   (tokens, segments, is_next)
        #
        # 其中:
        #   tokens:
        #       已经加入 <cls> 和 <sep> 的 token 序列。
        #
        #   segments:
        #       与 tokens 等长的 segment id。
        #
        #   is_next:
        #       True 表示句子 B 是句子 A 的真实下一句；
        #       False 表示句子 B 是随机抽取的句子。
        examples = []

        # 遍历每个段落，从段落内部构造 NSP 样本
        for paragraph in paragraphs:
            examples.extend(
                _get_nsp_data_from_paragraph(
                    paragraph,
                    paragraphs,
                    self.vocab,
                    max_len
                )
            )

        # 5. 在 NSP 样本基础上继续构造 MLM 数据
        #
        # 原来每个 example 是:
        #   (tokens, segments, is_next)
        #
        # _get_mlm_data_from_tokens(tokens, self.vocab) 返回:
        #   token_ids:
        #       经过 MLM 替换后的输入 token id 序列；
        #
        #   pred_positions:
        #       MLM 需要预测的位置；
        #
        #   mlm_pred_label_ids:
        #       这些位置对应的真实 token id 标签。
        #
        # 所以最终每个 example 变成:
        #   (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
        examples = [
            (
                _get_mlm_data_from_tokens(tokens, self.vocab)
                + (segments, is_next)
            )
            for tokens, segments, is_next in examples
        ]

        # 6. 对所有样本进行 padding
        #
        # token_ids 和 segments padding 到 max_len；
        # pred_positions、mlm_weights、mlm_labels padding 到 max_num_mlm_preds；
        # 同时生成 valid_lens 和 nsp_labels。
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels
        ) = _pad_bert_inputs(
            examples,
            max_len,
            self.vocab
        )

    def __getitem__(self, idx):
        """根据索引 idx 返回一条 BERT 预训练样本。

        返回:
            token_ids:
                shape: (max_len,)

            segments:
                shape: (max_len,)

            valid_len:
                标量，表示非 <pad> 的有效 token 数量。

            pred_positions:
                shape: (max_num_mlm_preds,)

            mlm_weights:
                shape: (max_num_mlm_preds,)

            mlm_labels:
                shape: (max_num_mlm_preds,)

            nsp_label:
                标量，表示 NSP 标签。
        """

        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx]
        )

    def __len__(self):
        """返回数据集中样本数量。"""

        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载 WikiText-2 数据集，并返回 BERT 预训练所需的数据迭代器和词表。

    参数:
        batch_size:
            每个 batch 中包含多少条样本。
            例如 batch_size=512，表示每次训练取 512 条 BERT 预训练样本。

        max_len:
            BERT 输入序列的最大长度。
            每条样本的 token_ids 和 segments 都会被 padding 或截断到 max_len。
            例如 max_len=64，则每条输入序列长度统一为 64。

    返回:
        train_iter:
            PyTorch DataLoader。
            每次迭代返回一个 batch，包含：
                token_ids,
                segments,
                valid_lens,
                pred_positions,
                mlm_weights,
                mlm_labels,
                nsp_labels

        train_set.vocab:
            根据 WikiText-2 训练集构建出的词表。
    """

    # WikiText-2 数据集所在目录
    # 该目录下应该有 wiki.train.tokens 文件
    data_dir = "../data/wikitext-2"

    # 读取 WikiText-2 原始文本，并处理成段落列表
    #
    # paragraphs 的结构大致是：
    # [
    #     ['sentence 1', 'sentence 2', 'sentence 3'],
    #     ['sentence 4', 'sentence 5'],
    #     ...
    # ]
    #
    # 每个 paragraph 是一个段落；
    # 每个 paragraph 内部是多个句子字符串。
    paragraphs = _read_wiki(data_dir)

    # 构造 WikiText-2 的 BERT 预训练数据集
    #
    # _WikiTextDataset 内部会完成：
    # 1. 分词 tokenize
    # 2. 构建词表 Vocab
    # 3. 构造 NSP 数据
    # 4. 构造 MLM 数据
    # 5. padding 到 max_len
    #
    # train_set 里包含的是已经处理好的 BERT 预训练样本集合。
    #
    # 每条样本包括：
    # token_ids        BERT输入token
    # segments         区分句子A/B
    # valid_len        有效长度
    # pred_positions   MLM预测位置
    # mlm_weights      MLM损失权重
    # mlm_labels       MLM真实标签
    # nsp_label        NSP真实标签
    #
    # 同时 train_set 还保存了：
    # vocab            词表
    train_set = _WikiTextDataset(paragraphs, max_len)

    # 使用 DataLoader 将 Dataset 包装成可迭代的小批量数据
    #
    # shuffle=True:
    #   每个 epoch 打乱数据顺序
    #
    # num_workers:
    #   用多个子进程加载数据，提高读取效率
    train_iter = torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=4
    )

    # 返回训练数据迭代器和词表
    return train_iter, train_set.vocab


if __name__ == "__main__":

    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_iter:
        print(
            tokens_X.shape,
            segments_X.shape,
            valid_lens_x.shape,
            pred_positions_X.shape,
            mlm_weights_X.shape,
            mlm_Y.shape,
            nsp_y.shape
        )
        break

    print(len(vocab))
