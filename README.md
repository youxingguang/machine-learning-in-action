机器学习实战笔记

前言

传统机器学习：有监督学习和无监督学习，代码实现参照 机器学习实战 这本书 (原书的代码下载地址：https://www.manning.com/books/machine-learning-in-action);

因为python 版本迭代，原书提供代码可能有些不兼容，这部分已重新手敲。(当前使用 python 3.8)


一、 经典算法

监督学习使用带标注的数据集；无监督学习不需要使用带标注的数据集。

1. 线性回归 (Linear Regression)
   
2. 逻辑回归 (Logistic Regression) ✅
   
3. K近邻 (KNN) ✅
  
4. 决策树 (Decision Tree) ✅
  
5. 随机森林 (Random Forest)
  
6. 梯度提升树 (GBDT)
   
7. 支持向量机 (SVM) ✅

   核方法
   
9. 朴素贝叶斯 (Naive Bayes) ✅

   极大似然估计
   
10. K均值 (K-Means) ✅
   
11. 主成分分析 (PCA) ✅
  
11. Apriori ✅
   
12. EM 算法
  

二、 集成学习

集成学习，通过构建并组合多个学习器来完成任务。学习器是学习算法的产物，可以由一个学习算法得到多个不同的学习器。所以集成学习并不强调多个算法。
两大主流思想：Bagging 和 Boosting。

13. XGBoost
14. LightGBM
15. CatBoost
16. AdaBoost ✅
17. DBSCAN
18. 层次聚类
19. t-SNE
20. LDA (主题模型)
    用于推测文档的主题分布。

三、 深度学习

21. 多层感知机 (MLP)
22. 卷积神经网络 (CNN)
23. 循环神经网络 (RNN)
24. Transformer
25. BERT
26. ResNet 
