import jieba
import re
import numpy as np
import jieba.posseg as psg
import networkx as nx
import pandas as pd


def get_stop_dict(file):
    content = open(file, encoding='utf-8')
    word_list = []
    for c in content:
        c = re.sub('\n|\r', '', c)
        word_list.append(c)
    return word_list


def get_data(path):
    t = open(path, encoding='utf-8')
    data = t.read()
    t.close()
    return data


def get_wordlist(text, maxn, synonym_words, stop_words):
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n', 'nz', 'vn']
    counts = {}

    text_seg = psg.cut(text)
    for word_flag in text_seg:
        word = word_flag.word
        if word_flag.flag in flag_list and len(word) > 1 and word not in stop_words:
            if word in synonym_origin:
                index = synonym_origin.index(word)
                word = synonym_new[index]
            counts[word] = counts.get(word, 0) + 1

    words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    words = list(dict(words).keys())[0:maxn]
    return words


def get_t_seg(topwords, text, synonym_words, stop_words):
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n', 'nz', 'vn']

    text_lines_seg = []
    text_lines = text.split('\n')
    for line in text_lines:
        t_seg = []
        text_seg = psg.cut(line)
        for word_flag in text_seg:
            word = word_flag.word
            if word_flag.flag in flag_list and len(word) > 1 and word not in stop_words:
                if word in synonym_origin:
                    word = synonym_new[synonym_origin.index(word)]
                if word in topwords:
                    t_seg.append(word)
        t_seg = list(set(t_seg))
        text_lines_seg.append(t_seg)
    return text_lines_seg


def get_comatrix(text_lines_seg, topwords):
    comatrix = pd.DataFrame(np.zeros([len(topwords), len(topwords)]), columns=topwords, index=topwords)
    for t_seg in text_lines_seg:
        for i in range(len(t_seg) - 1):
            for j in range(i + 1, len(t_seg)):
                comatrix.loc[t_seg[i], t_seg[i]] += 1
    for k in range(len(comatrix)):
        comatrix.iloc[k, k] = 0
    return comatrix


def get_net(co_matric, topwords):
    g = nx.Graph
    for i in range(len(topwords) - 1):
        word = topwords[i]
        for j in range(i + 1, len(topwords)):
            w = 0
            word2 = topwords[j]
            w = co_matric.loc[word][word2] + co_matric.loc[word2][word]
            if w > 0:
                g.add_edge(word, word2, weight=w)
    return g


# 文件路径
dic_file = '../co_occurrence_net/stop_dict/dict.txt'
stop_file = '../co_occurrence_net/stop_dict/stopwords.txt'
data_path = '../co_occurrence_net/data/data.txt'
synonym_file = '../co_occurrence_net/stop_dict/synonym_list.xlsx'
result_path = '../co_occurrence_net/result/word_cooccurrence.gexf'

# 读取文件
data = get_data(data_path)
stop_words = get_stop_dict(stop_file)
jieba.load_userdict(dic_file)
synonym_words = pd.read_excel(synonym_file)

# 数据处理
n_topwords = 200
topwords = get_wordlist(data, n_topwords, synonym_words, stop_words)
t_segs = get_t_seg(topwords, data, synonym_words, stop_words)


co_matrix = get_comatrix(t_segs, topwords)
co_net = get_net(co_matrix, topwords)

nx.write_gexf(co_net, '../co_occurrence_net/result/word_cooccurrence.gexf')
