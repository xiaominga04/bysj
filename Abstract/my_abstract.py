import torch
import lawrouge
import numpy as np

from typing import List, Dict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import datasets
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          BartForConditionalGeneration)

# 参数定义
batch_size = 4
epochs = 5
max_input_length = 512
max_target_length = 128
learning_rate = 1e-04


# 加载数据
dataset = load_dataset('json', data_files='nlpcc2017_clean.json', field='data')
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


# 数据处理
# 调整数据格式
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }


dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
print(type(dataset), dataset)

# 划分数据集
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
train_dataset, test_dataset = train_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
datasets = datasets.DatasetDict({"train": test_dataset, "validation": valid_dataset, "teat": test_dataset})


# 分词
def preprocess_function(examples):
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = datasets
tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True, remove_columns=["document", "summary", "id"])


# 定义批处理函数
def collate_fn(features: Dict):
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) for feature in features]

    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }


dataloader = DataLoader(tokenized_datasets["teat"], shuffle=False, batch_size=4, collate_fn=collate_fn)
batch = next(iter(dataloader))


# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-base-chinese")


# 模型训练
# 定义评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(pred.replace(" ", "")) for pred in decoded_labels]

    rouge = lawrouge.Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
    result = {key: value * 100 for  key, value in result.items()}
    return result


# 设置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="../results",
    num_train_epochs=epochs,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_steps=500,
    weight_decay=0.001,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    evaluation_strategy="steps",
    save_total_limit=3,
    generation_max_length=max_target_length,
    generation_num_beams=1,
    load_best_model_at_end=True,
    metric_for_best_model="rouge-1"
)


# 定义trainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=collate_fn,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# 训练
train_result = trainer.train(resume_from_checkpoint=True)
#print(trainer.evaluate(tokenized_datasets["validation"]))
#print(trainer.evaluate(tokenized_datasets["test"]))
trainer.save_model("results/best")


# 生成
model = BartForConditionalGeneration.from_pretrained("results/best")
test_examples = test_dataset["document"][:4]
inputs = tokenizer(
    test_examples,
    padding="max_length",
    truncation=True,
    max_length=max_input_length,
    return_tensors="pt"
)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)
outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_str = [s.replace(" ", "") for s in output_str]
print(output_str)
