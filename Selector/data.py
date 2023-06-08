import pandas as pd
import random
import json

data = pd.read_csv('../kg/train.tsv', sep='\t', on_bad_lines='skip')

# 正负样本构造
result = []
for i in range(len(data)):
    j = random.randint(0, len(data) - 1)
    nkeywords = data.values[j][2]
    nlabel = 0
    keywords = data.values[i][2]
    text = data.values[i][1]
    label = 1
    result.append({
        'keywords': keywords,
        'text': text,
        'label': label
    }
    )
    result.append({
        'keywords': nkeywords,
        'text': text,
        'label': nlabel
    }
    )


with open('./data/data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f)

