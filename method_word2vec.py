# -*- "coding: utf-8" -*-

from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding_filter.txt', binary=False)  # 加载腾讯开放的词向量集（已过滤，只包含vocabulary中的词）

with open("vocabulary_filter.txt", "r", encoding="utf-8") as f:
    vocabulary = f.read().split("\n")[:-1]

with open("method_word2vec.csv", "w", encoding="utf-8") as f:
    for first in vocabulary:
        for item in wv.most_similar(first, topn=3):  # 获得最相似的3个词语
            second, score = item
            f.write("{},{},{}\n".format(first, second, score))
