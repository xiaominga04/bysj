import Levenshtein
from Knowledge import knowledge_acquisition


# 文本编辑距离选取top10
def top_knowledge1(knowledge_list, keyword_list):
    scores = []
    for i in knowledge_list:
        scores.append((Levenshtein.distance(i, keyword_list) / len(i), i))
    scores = sorted(scores)
    return scores[0:min(10, len(scores) - 1)]


data = knowledge_acquisition.Keyword2Knowledge()
list = ['辛亥革命', '清政府']
keyword_list = '辛亥革命_清政府'
knowledge_list = data.get_knowledge_list(list)
print(top_knowledge1(knowledge_list, keyword_list))

