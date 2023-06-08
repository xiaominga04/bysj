from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import json
import numpy as np
'''
keywords = "染色_活性染料_匀染性_棉织物"
text = "双宫能团活性艳蓝GN和RN在固色浴中凝聚性小、骤染性小、匀染性好,且吸尽率和固色率高、提升性和重现性好,较好地克服了常用单乙烯砜型活性艳蓝" \
       "(C.I.B-19)的性能缺陷.该染料最适合70℃染色,与嫩黄Y-160或翠蓝B-21配伍拼染艳绿色或艳蓝色,可以大幅提高染色一等品率. 双官能团活性艳蓝的应用性能."
sentence = "[CLS]" + keywords + "[SEP]" + text + "[SEP]"

# 分词
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
sentence_token = tokenizer.tokenize(sentence)

max_len = 256
sentences_id = tokenizer.convert_tokens_to_ids(sentence_token)

for i in range(max_len):
    if i < len(sentences_id):
        continue
    sentences_id.append(0)


attention_mask = [1 if id > 0 else 0 for id in sentences_id]

sentences_id = torch.tensor(sentences_id)
attention_mask = torch.tensor(attention_mask)


model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.load_state_dict(torch.load('./result/result.pth'), strict=True)

batch_size = 1
device = 'cpu'

model.eval()

test_dataset = TensorDataset(sentences_id, attention_mask)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

model.eval()

for batch in test_dataloader:
    batch = tuple(data.to(device) for data in batch)
    inputs_ids, inputs_masks= batch
    with torch.no_grad():
        preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)
    preds = preds['logits'].detach().to('cpu').numpy()
    print(preds)
'''

def accuracy(labels, preds):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(preds)
    return acc

# 加载验证数据集
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
batch_size = 1
device = 'cpu'
max_len = 256

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.load_state_dict(torch.load('./result/result.pth'), strict=True)

data = [{
        'keywords': "染色_活性染料_匀染性_棉织物",
        'text': "双宫能团活性艳蓝GN和RN在固色浴中凝聚性小、骤染性小、匀染性好,且吸尽率和固色率高、提升性和重现性好,较好地克服了常用单乙烯砜型活性艳蓝" \
       "(C.I.B-19)的性能缺陷.该染料最适合70℃染色,与嫩黄Y-160或翠蓝B-21配伍拼染艳绿色或艳蓝色,可以大幅提高染色一等品率. 双官能团活性艳蓝的应用性能.",
        'label': 0
    },
{
        'keywords': "染色_活性染料_障碍_原因",
        'text': "双宫能团活性艳蓝GN和RN在固色浴中凝聚性小、骤染性小、匀染性好,且吸尽率和固色率高、提升性和重现性好,较好地克服了常用单乙烯砜型活性艳蓝" \
       "(C.I.B-19)的性能缺陷.该染料最适合70℃染色,与嫩黄Y-160或翠蓝B-21配伍拼染艳绿色或艳蓝色,可以大幅提高染色一等品率. 双官能团活性艳蓝的应用性能.",
        'label': 0
    },
    {'keywords': '基本药物_可获得性_障碍_原因', 'text': ' 在全党开展深入学习实践科学发展观活动,是党的十七大作出的一项重大战略部署.党中央对开展这项活动高度重视,决定先进行试点,取得经验后再自上而下分批展开. 最新理论成果重大战略思想——为什么要深入贯彻落实科学发展观', 'label': 0}

]


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
    print(preds)
    input_labels = input_labels.to('cpu').numpy()
    acc = accuracy(input_labels, preds)
    test_acc += acc
    steps += 1
    num += len(inputs_ids)

print("steps = ", steps)
print("test number = ", num)
print("test acc : {}".format(test_acc / steps))
