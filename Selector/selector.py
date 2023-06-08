import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR

from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup

from tqdm import tqdm, trange
import warnings
import json

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载数据集
with open('./data/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)[0:1000]

print(data[0], data[1])
# 构建Bert语料的输入格式
sentences = []
labels = []
for i in range(len(data)):

    sentences += ["[CLS]" + data[i]['keywords'] + "[SEP]" + data[i]['text'] + "[SEP]"]
    labels += [data[i]['label']]

print(sentences[0])
# 分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
sentence_tokens = [tokenizer.tokenize(sen) for sen in sentences]
print(sentence_tokens[0])

max_len = 256
sentences_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentence_tokens]

sentences_ids = pad_sequences(sentences_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')
attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentences_ids]

print(sentences_ids[0])
print(attention_mask[0])

# 数据集分离
X_train, X_eval, y_train, y_eval = train_test_split(sentences_ids, labels, test_size=0.2, random_state=666)
train_masks, eval_masks, _, _ = train_test_split(attention_mask, sentences_ids, test_size=0.2, random_state=666)


# 数据类型转换
X_train = torch.tensor(X_train)
X_eval = torch.tensor(X_eval)
y_train = torch.tensor(y_train)
y_eval = torch.tensor(y_eval)
train_masks = torch.tensor(train_masks)
eval_masks = torch.tensor(eval_masks)

# 数据打包
batch_size = 4
train_dataset = TensorDataset(X_train, train_masks, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataset = TensorDataset(X_eval, eval_masks, y_eval)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

# 模型训练
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

params = list(model.named_parameters())

EPOCHS = 4
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * EPOCHS)


def accuracy(labels, preds):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(preds)
    return acc


train_loss = []
for i in tqdm(range(EPOCHS), desc='Epoch'):
    model.train()
    tr_loss = 0
    tr_examples = 0
    tr_steps = 0

    for i, batch_data in enumerate(train_dataloader):
        '''
        if i == 0:
            print(len(train_dataloader))
        print(i)
        '''
        batch_data = tuple(data.to(device) for data in batch_data)
        inputs_ids, inputs_masks, input_labels = batch_data
        optimizer.zero_grad()
        outputs = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks, labels=input_labels)
        loss = outputs['loss']
        train_loss.append(loss.item())
        tr_loss += loss.item()
        tr_examples += inputs_ids.size(0)
        tr_steps += 1
        loss.backward()
        optimizer.step()
        scheduler.step()

    print("Training loss : {}".format(tr_loss / tr_steps))

    model.eval()
    eval_acc = 0.0, 0.0
    eval_steps, eval_examples =0.0, 0.0

    for batch in eval_dataloader:
        batch = tuple(data.to(device) for data in batch_data)
        inputs_ids, inputs_masks, input_labels = batch
        with torch.no_grad():
            preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
        preds = preds['logits'].detach().to('cpu').numpy()
        labels = input_labels.to('cpu').numpy()
        eval_acc += accuracy(labels, preds)
        eval_steps += 1

    print("Eval Accuracy : {}".format(eval_acc / eval_steps))
    print("\n\n")

plt.figure(figsize=(12, 10))
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss)
plt.show()

# model.save_pretrained('./result')
# torch.save(model.state_dict(), "./result/result.pth")

# 加载验证数据集
with open('./data/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)[1000:1500]


# 构建Bert语料的输入格式
sentences = []
labels = []
for i in range(len(data)):
    sentences += ["[CLS]" + data[i]['keywords'] + "[SEP]" + data[i]['text'] + "[SEP]"]
    labels += [data[i]['label']]

sentence_tokens = [tokenizer.tokenize(sen) for sen in sentences]
sentences_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentence_tokens]
sentences_ids = pad_sequences(sentences_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')
attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentences_ids]

sentences_ids = torch.tensor(sentences_ids)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(labels)

test_dataset = TensorDataset(sentences_ids, attention_mask, labels)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

model.eval()
test_loss, test_acc = 0.0, 0.0
steps = 0
num = 0

for batch in test_dataloader:
    batch = tuple(data.to(device) for data in batch)
    inputs_ids, inputs_masks, input_labels = batch
    with torch.no_grad():
        preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
    preds = preds['logits'].detach().to('cpu').numpy()
    input_labels = input_labels.to('cpu').numpy()
    acc = accuracy(input_labels, preds)
    test_acc += acc
    steps += 1
    num += len(inputs_ids)

print("steps = ", steps)
print("test number = ", num)
print("test acc : {}".format(test_acc / steps))
