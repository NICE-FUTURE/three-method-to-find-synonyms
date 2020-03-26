# -*- "coding: utf-8" -*-

import numpy as np

class Cosine(object):

    def __init__(self, n_recommendation, step=3000):
        self.n_recommendation = n_recommendation
        self.step = step

    def _cal_cosine(self, X, Y):
        '''
        计算余弦相似度，X 为局部数据，Y 为完整数据
        Args:
            X: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
            Y: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
        Returns:
            cosine: ndarray 第i行第j列的值是第i个元素和第j个元素之间的相似度
        '''
        numerator = X.dot(Y.T)  # 向量 a 乘以向量 b (分子)
        norm_X = np.linalg.norm(X, axis=1)  # 每行一个范数，结果为一维 ndarray 数据
        norm_Y = np.linalg.norm(Y, axis=1)  # 每行一个范数，结果为一维 ndarray 数据
        cosine = numerator /(10e-20+np.expand_dims(norm_X, axis=1)) /(10e-20+np.expand_dims(norm_Y, axis=0))  # 为防止零作除数做的处理，这将导致完全相同的两个向量相似度不能完全等于1
        return cosine

    def cal_similarity(self, X, Y):
        '''
        组织数据分块计算相似度
        Args:
            X: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
            Y: ndarray  行数m表示有m个元素，列数n表示每个元素用n个特征表示
        Returns:
            indices: list  每个元素为相似度矩阵的一行从大到小排列的索引位置 [[idx1, idx2, ...], ...]
            similarities: list  每一个元素为相似度矩阵的一行从大到小排列的值 [[sim1, sim2, ...], ...]
        '''
        pointer = 0
        indices = []
        similarities = []
        while pointer <= X.shape[0]:
            similarity = self._cal_cosine(X[pointer:pointer+self.step], Y)
            indices += np.argsort(similarity, axis=1)[:, ::-1][:, :self.n_recommendation].tolist()
            similarities += np.sort(similarity, axis=1)[:, ::-1][:, :self.n_recommendation].tolist()
            pointer += self.step
        return indices, similarities


'''
if __name__ == "__main__":
    matrix = np.array([[0.2223, 0.1111, 1, 0.322, 0.00001], [0.5, 0.1, 1, 0, 1], [0.2, 0.1, 0.99999, 0.322, 0]])
    cosine = Cosine(n_recommendation=2)
    indices, similarities = cosine.cal_similarity(matrix)
    print(["{}:{}".format(index, similarity) for index, similarity in zip(indices, similarities)])
'''
