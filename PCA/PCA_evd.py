# 主成分分析  基于对样本协方差 特征值分解

import numpy as np

"""
用户指定  降到k维度
"""
def pca_via_evd(X,k):

    #1.数据规范化处理
    mean=np.mean(X,axis=0)
    X_centered=X-mean  # X (n,p)

    #2.计算协方差  分母n-1 保证无偏估计
    n=X.shape[0] #样本个数  n
    covariance_matrix=np.dot(X_centered.T,X_centered)/(n-1)

    #3.对协方差矩阵 进行特征值分解
    eig_vals,eig_vecs=np.linalg.eig(covariance_matrix)

    #4.对特征值进行降序 排序，并找到对应的特征向量
    sorted_indices=np.argsort(eig_vals)[::-1] #获取降序排列的索引
    sorted_eig_vals=eig_vals[sorted_indices]
    sorted_eig_vecs=eig_vecs[:,sorted_indices]

    #5.提取前k个 主成分的方向 （构建投影矩阵）
    W=sorted_eig_vecs[:,:k]  #因为特征值列可能不止k 所以:k 取到

    #6.利用投影空间 计算主成分 Y=W^TX
    Y=np.dot(X_centered,W)



    #计算 方差贡献率
    variance_ratio=sorted_eig_vals/np.sum(sorted_eig_vals)
    k_variance_ratio=np.cumsum(variance_ratio[:k])#取前k个 逐项后累加

    return Y,W,k_variance_ratio

#测试
np.random.seed(42)
X_test=np.random.rand(100,4) #100个样本  4个特征
k=2
Y,W,ratio=pca_via_evd(X_test,k)

print(f'降维后的数据形状{Y.shape}')
print(f'前{k}个主成分的方差比例{ratio}')