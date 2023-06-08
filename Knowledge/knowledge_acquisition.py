import pandas as pd
from collections import defaultdict


# 获取关键词和科学知识的链接关系
class Keyword2Knowledge(object):
    def __init__(self):
        self.keyword = set()
        self.k2k = defaultdict(list)

        def structure():
            data1 = pd.read_csv('../kg/train.tsv', sep='\t')
            data2 = pd.read_csv('../kg/test.tsv', sep='\t')
            data3 = pd.read_csv('../kg/dev.tsv', sep='\t')
            data4 = pd.read_csv('../kg/csl_camera_readly.tsv', sep='\t', on_bad_lines='skip')
            for i in range(len(data1)):
                keywords = data1.values[i][2].split('_')
                for j in range(len(keywords)):
                    self.keyword.add(keywords[j])
                    self.k2k[keywords[j]].append(data1.values[i][1])
            for i in range(len(data2)):
                keywords = data2.values[i][2].split('_')
                for j in range(len(keywords)):
                    self.keyword.add(keywords[j])
                    self.k2k[keywords[j]].append(data2.values[i][1])
            for i in range(len(data3)):
                keywords = data3.values[i][2].split('_')
                for j in range(len(keywords)):
                    self.keyword.add(keywords[j])
                    self.k2k[keywords[j]].append(data3.values[i][1])
            for i in range(len(data4)):
                keywords = data4.values[i][2].split('_')
                for j in range(len(keywords)):
                    self.keyword.add(keywords[j])
                    self.k2k[keywords[j]].append(data4.values[i][1])

        structure()

    def add_knowledge(self, path):
        data = pd.read_csv(path, sep='\t')
        for i in range(len(data)):
            keywords = data.values[i][2].split('_')
            for j in range(len(keywords)):
                self.keyword.add(keywords[j])
                self.k2k[keywords[j]].append(data.values[i][1])
        return

    def get_knowledge(self, keyword):
        return self.k2k[keyword]

    def get_knowledge_list(self, keyword_list):
        knowledge = []
        for i in keyword_list:
            knowledge += self.get_knowledge(i)
        for i in knowledge:
            print(i)
        return knowledge
