import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib


# 分词
data = open('../co_occurrence_net/data/data.txt', 'r', encoding='utf-8')
lines = []
for line in data:
    temp = jieba.lcut(line)
    words = []
    for i in temp:
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)

model = Word2Vec(lines, vector_size=20, window=2, min_count=3, epochs=7, negative=10)

print(model.wv.most_similar('环境', topn=20))

