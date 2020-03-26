# -*- "coding: utf-8" -*-

import synonyms
import numpy as np
from cosine import Cosine

cosine = Cosine(n_recommendation=4)

with open("vocabulary_filter.txt", "r", encoding="utf-8") as f:
    vocabulary = f.read().split()[:-1]

vectors = []
for word in vocabulary:
    try:
        vectors.append(synonyms.v(word))  # 使用 synonyms 获得词向量
    except:
        pass

vectors = np.array(vectors)

indices, similarities = cosine.cal_similarity(vectors, vectors)  # 调用cosine模块计算余弦相似度

with open("method_synonyms.csv", "w", encoding="utf-8") as f:
    for nrow, row in enumerate(indices):
        for ncol, col in enumerate(row):
            if ncol == 0:  # 跳过自身
                continue
            f.write("{},{},{}\n".format(vocabulary[nrow], vocabulary[col], similarities[nrow][ncol]))
