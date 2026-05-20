# logistic 回归

"""
   叫回归，但是是分类

   处理输入 x
   到中间 z=w^tx+b 对数几率
   将z 映射到 0-1 y=1/(1+e^(-z))

    算法训练 找到合适的w
    完成分类

"""
from numpy import*
import matplotlib.pyplot as plt


"""
数据处理 z=w^t x+b  将b处理进 w_0, 后面以w_1,w_2 对应特征
    假设 数据集： x_1 x_2  类标签
    x=(1,x_1,x_2) dataMat
    
"""
def loadDataSet(filepath):
    dataMat=[];labelMat=[]
    fr=open(filepath)
    #测试解  每行 点(x_1,x_2) 类标签
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


# inX = z
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升 求 w
"""
   w 沿着梯度方向 上升 
   w:=w+\alpha \nabla f(w)  学习率*梯度值
      \nabla f(w)=X^T(y-h) 
        h=1/(1+e^(-z)) 
"""

# 老的mat  A*B  在numpy array里 A@B
def gradAscent(dataMatIn,classLabels):
    dataMatrix=array(dataMatIn) # 转换为numpy 矩阵
    labelMat=array(classLabels).reshape(-1,1)
    m,n=shape(dataMatrix)
    alpha=0.001 #学习率
    maxCycles=500 #最大循环500
    weights=ones((n,1))

    for k in range(maxCycles):
        h=sigmoid(dataMatrix@weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.T@error
    return weights

#随机梯度上升
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    dataMatrix = array(dataMatrix)
    classLabels = array(classLabels)
    alpha=0.01
    weights=ones(n)
    history=[]
    for i in range(m):
        h=sigmoid(dataMatrix[i]@weights)
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
        history.append(weights.copy())
    return weights,array(history)

"""
比较 gradAscent 和 stocGradAscent0
   直接比较结果,随机梯度上升吃亏，因为梯度上升在整体样本循环了500次
   比较可靠的办法看算法是否收敛，
"""

#画出随机梯度 历史w的变化
def plotWeights(history):
    history = array(history)

    plt.figure(figsize=(10, 6))

    plt.plot(history[:, 0], label='w0')
    plt.plot(history[:, 1], label='w1')
    plt.plot(history[:, 2], label='w2')

    plt.xlabel('iteration')
    plt.ylabel('weight value')

    plt.legend()

    plt.show()


# 画出数据集 和 logistic回归的最佳拟合直线
def plotBestFit(wei,filepath):
    weights=wei
    dataMat,labelMat=loadDataSet(filepath)
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x=arange(-3.0,3.0,0.1) #坐标轴 精度
    # 0=w_0 x_0 +w_1 x_1 +w_2 x_2
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()



filepath=r"E:\机器学习笔记\logistic_regression\testSet.txt"
dataInput,classLabels=loadDataSet(filepath)
weights,history=stocGradAscent0(dataInput,classLabels)
plotWeights(history)
#plotBestFit(weights,filepath)

