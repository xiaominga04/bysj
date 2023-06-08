from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

#添加pad
#tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

print(tokenizer)

#编码试算
tokenizer.batch_encode_plus([
    'hide new secretions from the parental units',
    'contains no wit , only labored gags'
])

#%%

from datasets import load_dataset, concatenate_datasets


def get_dataset():
    #加载数据
    dataset = load_dataset('imdb')

    #重新切分数据集
    dataset = concatenate_datasets(
        [dataset['train'], dataset['test'], dataset['unsupervised']])

    dataset = dataset.train_test_split(test_size=0.01, seed=0)

    #采样,数据量太大了跑不动
    dataset['train'] = dataset['train'].shuffle(0).select(range(80000))
    dataset['test'] = dataset['test'].shuffle(0).select(range(200))

    #分词
    def f(data):
        #移除<br/>
        for i in range(len(data['text'])):
            data['text'][i] = data['text'][i].replace('<br /><br />', ' ')

        data = tokenizer.batch_encode_plus(data['text'])

        return data

    dataset = dataset.map(f,
                          batched=True,
                          num_proc=4,
                          batch_size=1000,
                          remove_columns=['text', 'label'])

    #过滤掉太短的句子
    def f(data):
        return [sum(i) >= 25 for i in data['attention_mask']]

    dataset = dataset.filter(f, batched=True, num_proc=4, batch_size=1000)

    #拼合句子到统一的长度
    def f(data):
        block_size = 512

        #展平数据
        input_ids = []
        for i in data['input_ids']:
            input_ids.extend(i)

        #切断数据
        data = {'input_ids': [], 'attention_mask': []}
        for i in range(len(input_ids) // block_size):
            block = input_ids[i * block_size:i * block_size + block_size]
            data['input_ids'].append(block)
            data['attention_mask'].append([1] * block_size)

        #设置labels
        data['labels'] = data['input_ids'].copy()

        return data

    dataset = dataset.map(
        f,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    return dataset


#直接使用我处理好的数据集
dataset = get_dataset()


#%%

import torch
from transformers.data.data_collator import default_data_collator

#数据加载器
loader = torch.utils.data.DataLoader(
    dataset=dataset['train'],
    batch_size=8,
    collate_fn=default_data_collator,
    shuffle=True,
    drop_last=True,
)

for i, data in enumerate(loader):
    break

len(loader), data

#%%

from transformers import AutoModelForCausalLM, GPT2Model, PreTrainedModel, PretrainedConfig

#加载模型
#model = AutoModelForCausalLM.from_pretrained('gpt2')


#定义下游任务模型
class Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.pretrained = GPT2Model.from_pretrained('gpt2')
        self.fc = torch.nn.Linear(768, tokenizer.vocab_size, bias=False)

        #加载预训练模型的参数
        parameters = AutoModelForCausalLM.from_pretrained('gpt2')
        self.fc.load_state_dict(parameters.lm_head.state_dict())

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        logits = self.pretrained(input_ids=input_ids,
                                 attention_mask=attention_mask)
        logits = logits.last_hidden_state

        logits = self.fc(logits)

        shift_logits = logits[:, :-1].flatten(end_dim=1)
        shift_labels = labels[:, 1:].flatten()

        loss = self.criterion(shift_logits, shift_labels)

        return {
            'loss': loss,
            'logits': logits,
        }


model = Model(PretrainedConfig())

#统计参数量
print(sum(i.numel() for i in model.parameters()) / 10000)

with torch.no_grad():
    out = model(**data)

out['loss'], out['logits'].shape

#%%

def generate(text):

    def generate_loop(data):
        with torch.no_grad():
            out = model(**data)

        #取最后一个字
        #[5, b, 50257]
        out = out['logits']
        #[5, 50257]
        out = out[:, -1]

        #第50大的值,以此为分界线,小于该值的全部赋值为负无穷
        #[5, 50257] -> [5, 50]
        topk_value = torch.topk(out, 50).values
        #[5, 50] -> [5] -> [5, 1]
        topk_value = topk_value[:, -1].unsqueeze(dim=1)

        #赋值
        #[5, 50257]
        out = out.masked_fill(out < topk_value, -float('inf'))

        #根据概率采样,无放回,所以不可能重复
        #[5, 50257] -> [5, 1]
        out = out.softmax(dim=1)
        out = out.multinomial(num_samples=1)

        data['input_ids'] = torch.cat([data['input_ids'], out], dim=1)
        data['attention_mask'] = torch.ones_like(data['input_ids'])
        data['labels'] = data['input_ids'].clone()

        if data['input_ids'].shape[1] >= 30:
            return data

        return generate_loop(data)

    #重复5遍
    data = tokenizer.batch_encode_plus([text] * 5, return_tensors='pt')
    data['labels'] = data['input_ids'].clone()

    data = generate_loop(data)

    for i in range(5):
        print(i, tokenizer.decode(data['input_ids'][i]))


generate('I love this')

#%%

from transformers import AdamW
from transformers.optimization import get_scheduler


#训练
def train():
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)

    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(device)
        out = model(**data)
        loss = out['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()
        model.zero_grad()

        if i % 50 == 0:
            labels = data['labels'][:, 1:]
            out = out['logits'].argmax(dim=2)[:, :-1]

            accuracy = (labels == out).sum().item() / labels.numel()

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            print(i, loss.item(), lr, accuracy)

    model.to('cpu')


#%%

#直接使用我训练好的模型
train()

generate('I love this')
