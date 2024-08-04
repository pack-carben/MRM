import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


class NMRMEM:
    def __init__(self, data, n_components, tol=1e-3, max_iter=100,random_state=123):
        self.data = data.copy() #居然在这里碰见了对象问题......必须复制
        self.Y = self.data[:, 0]
        self.X = data.copy()
        self.X[:, 0] = 1

        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_samples, self.n_features = data.shape
        self.residuals = np.zeros((self.n_samples, self.n_components))
        self.probability = None
        self.z = np.zeros((self.n_samples,self.n_components)) #随机变量Z期望矩阵
        self.sum_z = None
        self.random_state = random_state
        # 初始化其他属性
        self.labels = None  # 存储聚类标签
        self.cluster_centers = None  # 存储聚类中心
        self.beta = None  # 存储回归系数
        self.sigma = None  # 存储回归残差标准差

    def initialization(self):
        X=self.data[:, 1:]
        Y=self.data[:, 0]
        # 执行K均值聚类
        kmeans = KMeans(n_clusters=self.n_components, tol=self.tol, max_iter=self.max_iter,
                        random_state=self.random_state)
        kmeans.fit(self.data)

        # 获取聚类标签和质心
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        self.probability = np.bincount(self.labels) / float(self.n_samples)
        self.probability = np.matrix(self.probability)   #转换后默认行向量矩阵
        # 初始化存储beta和sigma的列表
        betas = []
        sigmas = []

        # # 在每个聚类组中进行线性回归
        for label in range(self.n_components):
            # 筛选属于当前聚类的数据
            mask = (self.labels == label)
            X_cluster = X[mask]
            Y_cluster = Y[mask]

            # 执行线性回归
            model = LinearRegression(fit_intercept=True)
            model.fit(X_cluster, Y_cluster)

            # 提取回归系数和残差标准差
            beta = np.append(model.intercept_, model.coef_)
            sig = np.std(model.predict(X_cluster) - Y_cluster)

            # 存储结果
            betas.append(beta)
            sigmas.append(sig)
        # 在每个聚类组中进行线性回归
        # for label in range(self.n_components):
        #     # 筛选属于当前聚类的数据
        #     mask = (self.labels == label)
        #     X_cluster = X[mask]
        #     Y_cluster = Y[mask]
        #
        #     # 进行分位数筛选，保留95%分位数内的数据
        #
        #     lower_bound = np.percentile(Y_cluster, 2.5)
        #     upper_bound = np.percentile(Y_cluster, 97.5)
        #     filter_mask = (Y_cluster >= lower_bound) & (Y_cluster<= upper_bound)
        #     X_cluster_filtered = X_cluster[filter_mask]
        #     Y_cluster_filtered = Y_cluster[filter_mask]
        #
        #     # 执行线性回归
        #     model = LinearRegression(fit_intercept=True)
        #     model.fit(X_cluster_filtered, Y_cluster_filtered)
        #
        #     # 提取回归系数和残差标准差
        #     beta = np.append(model.intercept_, model.coef_)
        #     sig = np.std(model.predict(X_cluster_filtered) - Y_cluster_filtered)
        # 转换为数组形式
        self.beta = np.array(betas)
        self.sigma = np.array(sigmas)
        for i in range(self.n_samples):
            for k in range(self.n_components):
                self.residuals[i, k] = self.Y[i] - np.dot(self.beta[k,:],self.X[i,:])
        return self.beta, self.sigma

    def expectation(self):
        if self.probability is None or self.sigma is None:
            raise ValueError("Probability and sigma must be computed before expectation.")

        for i in range(self.n_samples):
            for k in range(self.n_components):
                #test=self.probability[k] * norm.pdf(self.residuals[i, k], loc=0, scale=self.sigma[k])
                #print(type(self.probability[k] * norm.pdf(self.residuals[i, k], loc=0, scale=self.sigma[k])))
                #初始化时不是matrix
                self.z[i, k] = self.probability[:,k] * norm.pdf(self.residuals[i, k], loc=0, scale=self.sigma[k])

                #self.z[i, k]=norm.pdf(self.residuals[i, k], loc=0, scale=self.sigma[k])
            # 归一化每个样本对每个聚类的期望值
            self.z[i, :] /= np.sum(self.z[i, :])

    def maximization(self):
        self.sum_z = np.sum(self.z, axis=0, keepdims=True)
        self.probability = self.sum_z / self.n_samples

        diag_vector = np.sum(self.X * self.X, axis=1)
        for k in range(self.n_components):
            for m in range(self.n_features):
                numerator = np.sum(self.z[:, k] * self.Y * self.X[:, m], axis=0)

                denominator = np.sum(self.z[:, k] * diag_vector, axis=0)
                # 计算分子：对每个样本的特征和响应变量进行加权乘积
                #numerator = np.sum(self.z[:, k] * self.Y * self.X[:, m])

                # 计算分母：对每个样本的特征的加权平方和
                #denominator = np.sum(self.z[:, k] * self.X[:, m] ** 2)
                self.beta[k,m] = numerator / denominator

            self.residuals[:, k] = self.Y - np.dot(self.X, self.beta[k])
            std=np.sum(self.z[:,k]*self.residuals[:, k]*self.residuals[:, k],axis=0)
            self.sigma[k] = math.sqrt(std/self.sum_z[:, k])

    def fit(self):
        self.initialization()
        for i in range(self.max_iter):
            self.expectation()
            self.maximization()
        return self.beta, self.sigma


# 示例用法
if __name__ == "__main__":
    # 示例数据

    data = pd.read_csv('test.csv')
    data=data.to_numpy()
    # 创建NMRMEM对象
    model = NMRMEM(data, n_components=3,max_iter=100)

    # 执行聚类和线性回归
    beta, sigma = model.fit()
    # 设置 NumPy 的打印选项
    np.set_printoptions(formatter={'float': '{:0.2f}'.format})
    # 输出结果
    print("Beta:")
    print(beta)

    print("Sigma:")
    print(sigma)
    print("Probability:")
    print(model.probability)

