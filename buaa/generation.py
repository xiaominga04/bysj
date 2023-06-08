from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

print(tokenizer)

tokenizer.batch_encode_plus([
    'hide new secretions from the parental units',
    'contains no wit, only labored gags'
])


from datasets import load_dataset, load_from_disk, concatenate_datasets


dataset = load_dataset('imdb')

dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['unsupervised']])

dataset = dataset.train_test_split(test_size=0.01, seed=0)

dataset['train'] = dataset['train'].shuffle(0).select(range(80000))
dataset['test'] = dataset['test'].shuffle(0).select(range(200))


def f(data):
    for i in range(len(data['text'])):
        data['text'][i] = data['text'][i].replace('<br /><br />', ' ')
    data = tokenizer.batch_encode_plus(data['text'])
    return data


dataset = dataset.map(f, batched=True, num_proc=4, batch_size=1000, remove_columns=['text', 'label'])


def f(data):
    return [sum(i) >= 25 for i in data['attention_mask']]


dataset = dataset.filter(f, batched=True, num_proc=4, batch_size=1000)

def f(data):
    block_size = 512

    input_ids = []
    for i in data['input_ids']:
        input_ids.extend(i)

    data = {'input_ids':[], 'attention_mask':[]}
    for i in  range(len(input_ids) // block_size):
        block = input_ids[i * block_size : i * block_size + block_size]
        data['input_ids'].append(block)
        data['attention_mask'].append([1] * block_size)

    data['labels'] = data['input_ids'].copy()

    return data


dataset = dataset.map(f, batched=True, batch_size=1000, num_proc=4)

print(dataset)
