import torch
from datasets import load_dataset


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __int__(self, split):
        dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)

        # 过滤数据集
        def f(data):
            return len(data['text']) > 30

        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        return text


dataset = Dataset('train')
print(type(dataset), dataset[0])

# 加载tokenizer
from transformers import BertTokenizer

token = BertTokenizer.from_pretrained('bert-base-chinese')
print(token)


# 定义批处理函数
def collate_fn(data):
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=data,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt',
        return_length=True
    )
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    labels = input_ids[:, 15].reshape(-1).clone
    input_ids[:, 15] = token.get_vocab()[token.mask_token]

    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = torch.utils.data.DataLoader(
    data=dataset,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True
)

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break

print(len(loader))
print(token.decode(input_ids[0]))
print(token.decode(labels[0]))

# 加载bert中文模型
from transformers import BertModel

pretrained = BertModel.from_pretrained('bert-base-chinese')

for param in pretrained.parameters():
    param.requiers_grad_(False)

out = pretrained(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids
)