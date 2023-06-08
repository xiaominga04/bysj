from transformers import BertTokenizer


# 读入数据集，并编码
def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):

            line = line[:-1].split('\t')

            if len(line) == 3:
                text = line[0] + SEP_TOKEN + line[1]
                label = line[2]
            else:
                text, label = line[0], line[1]

            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
            tgt_in = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(label) + [SEP_TOKEN])
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            if len(tgt_in) > args.tgt_seq_length:
                tgt_in = tgt_in[: args.tgt_seq_length]
            tgt_out = tgt_in[1:] + [PAD_ID]

            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            while len(tgt_in) < args.tgt_seq_length:
                tgt_in.append(PAD_ID)
                tgt_out.append(PAD_ID)

            dataset.append((src, tgt_in, tgt_out, seg))

    return dataset