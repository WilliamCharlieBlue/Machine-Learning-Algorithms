import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim == 2, "The dimension of X must be 2"
        # 有多少列(特征)就计算多少个均值/方差
        self.mean_ = np.array([np.mean(X[:i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:i]) for i in range(X.shape[1])])

        return self

    def tranform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and std_"
        # 先预设放回矩阵为浮点型
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX


class MinMaxScaler:

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """根据训练数据集X获得数据的最小值和最大值"""
        assert X.ndim == 2, "The dimension of X must be 2"
        # 有多少列(特征)就计算多少个最大值/最小值
        self.min_ = np.array([np.min(X[:i]) for i in range(X.shape[1])])
        self.max_ = np.array([np.max(X[:i]) for i in range(X.shape[1])])

        return self

    def tranform(self, X):
        """将X根据这个StandardScaler进行最值归一化处理"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.min_ is not None and self.max_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.min_) and X.shape[1] == len(self.max_), \
            "the feature number of X must be equal to min_ and max_"
        # 先预设放回矩阵为浮点型
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.min_[col]) / (self.max_[col] - self.min_[col])
        return resX