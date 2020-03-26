# -*- "coding: utf-8" -*-

from cosine import Cosine
from bert_serving.client import BertClient

bc = BertClient()
cosine = Cosine(n_recommendation=4)

with open("vocabulary_filter.txt", "r", encoding="utf-8") as f:
    vocabulary = f.read().split()[:-1]

vectors = bc.encode(vocabulary)  # 使用 Bert 获得所有词语的词向量
indices, similarities = cosine.cal_similarity(vectors, vectors)  # 调用cosine模块计算余弦相似度

with open("method_bert.csv", "w", encoding="utf-8") as f:
    for nrow, row in enumerate(indices):
        for ncol, col in enumerate(row):
            if ncol == 0:  # 跳过自身
                continue
            f.write("{},{},{}\n".format(vocabulary[nrow], vocabulary[col], similarities[nrow][ncol]))
