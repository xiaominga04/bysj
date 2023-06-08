import pandas as pd
import re

t = open('../co_occurrence_net/data/data.txt', encoding='utf-8', mode='w')

data = pd.read_csv('../data.csv', sep=',', on_bad_lines='skip')

for i in range(len(data)):
    if data.values[i][0] == "环境":
        l = eval(data.values[i][1])
        for j in l:
            t.write(re.sub('，|,|。', '\n', j))
        break
t.close()
