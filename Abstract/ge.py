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

max_input_length = 512
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = BartForConditionalGeneration.from_pretrained("results/best")
test_examples = ["辛亥革命在推翻了清政府的腐朽统治之后,以孙中山为首的革命党人以一个全新的政治姿态走上中国政治舞台"]
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