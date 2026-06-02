# 主成分分析  基于奇异值svd分解

import numpy as np

def pca_via_svd(X,k):

    mean=np.mean(X,axis=0)
    X_centered=X-mean

    #SVD分解
    # X_centered=U*S*Vt  截断(假设有r个,后面截取k个)
    U,S,Vt=np.linalg.svd(X_centered,full_matrices=False)

    """
    SVD 返回的S是按降序排列的奇异值 
    Vt 的行向量 也是按降序对应主成分方向
    """

    #提取前k个
    W=Vt[:k,:].T

    # 数据投影 得到主成分
    Y=np.dot(X_centered,W)

    #由奇异值计算特征值 方差贡献率
    n=X.shape[0]
    eig_vals=(S**2)/(n-1)
    variance_ratio=eig_vals/np.sum(eig_vals)
    k_variance_ratio=variance_ratio[:k]

    return Y,W,k_variance_ratio

#测试
np.random.seed(42)
X_test=np.random.rand(100,4) #100个样本  4个特征
k=2
Y,W,k_variance_ratio=pca_via_svd(X_test,k)
print(f'降维后的数据形状{Y.shape}')
print(f'前{k}个主成分的方差比例{k_variance_ratio}')