#基于 决策树 的自适应 提升方法  Adaboost

import numpy as np

#树桩分类  简单的决策树
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    对 第i列变量 使用 <threshVal 判定
    :param dataMatrix: 样本集
    :param dimen: 维度，指示 第 i 列
    :param threshVal: 阈值
    :param threshIneq: 不等关系 取{<, >}中一个
    :return:
    retArray 预测分类 初始全1, 现在采用< 分类, 若小于阈值 分类-1;同理 当采用 >分类, 大于阈值 分类-1
    """
    retArray=np.ones(dataMatrix.shape[0])
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray
#简化版本
def buildStump(dataArr,classLabel,D):
    dataMatrix=np.array(dataArr)
    labelMat=np.array(classLabel).flatten()
    m,n=dataMatrix.shape
    numSteps=10.0 #将[min,max] 分为10段

    bestStump={} #记录本轮 最优秀的树桩(对哪一列进行怎样的操作）
    bestClasEst=np.zeros(m) #每个样本的分类结果
    minError=float('inf') # m个样本加权误差 ->最佳

    for i in range(n):
        # 第一层循环 维度（有n列）


        #取出 该列的最大 最小值
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()

        stepSize=(rangeMax-rangeMin)/numSteps #分成10段,每段的分度
        for j in range(-1,int(numSteps)+1):

            #lt less than <;    gt >  调整符号
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize) #调整阈值
                predictVals=stumpClassify(dataMatrix,i,threshVal,inequal) #依据i列 分类的m个结果

                errArr=np.ones(m) #标记 m个样本的错误
                errArr[predictVals==labelMat]=0 #正确 0
                weightedError=D@errArr #根据 样本的权重，加权求和 累计误差
                print(f"分割第 {i} 维度, 阈值: {threshVal:.2f}, 阈值不等式: {inequal}, 加权误差: {weightedError:.3f}")

                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    dataArr=np.array(dataArr)
    classLabels=np.array(classLabels).flatten() #保持一维
    weakClassArr=[] #记录分类器的权重
    m=dataArr.shape[0]
    D=np.ones(m)/m  #初始时 所有样本 权重取值相等
    aggClassEst=np.ones(m)
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        alpha=float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump.copy())
        # 因为乘 -alpha 和 alpha
        expon=-1*alpha*classLabels*classEst
        D=D*np.exp(expon)
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        aggErrors=(np.sign(aggClassEst)!=classLabels).astype(int)
        errorRate=aggErrors.sum()/m
        print(f"第 {i + 1} 轮迭代，当前训练集错误率: {errorRate:.4f}")
        if errorRate==0.0:
            break
    print("最终分类器：")
    print(weakClassArr)
    return weakClassArr

# 测试函数
def adaClassify(datToClass,classifierArr):
    dataMatrix=np.array(datToClass)
    m=dataMatrix.shape[0]
    aggClassEst=np.zeros(m) #记录m个样本的加权分类 得分
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                               classifierArr[i]['thresh'],\
                               classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return np.sign(aggClassEst) # 转化 >0 +1, <0 -1

#分类标签 保证{-1,1}
def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[];labelMat=[] #新建两个空列表
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#加载 训练集
fileName1=r"E:\机器学习笔记\AdaBoost\horseColicTraining2.txt"
dataMat,labelMat=loadDataSet(fileName1)

#加载测试集
fileName2=r"E:\机器学习笔记\AdaBoost\horseColicTest2.txt"
dataMat2,labelMat2=loadDataSet(fileName2)
datToClass=np.column_stack((dataMat2,labelMat2)) #按列拼接起来

weakClassArr=adaBoostTrainDS(dataMat,labelMat)

predictions = adaClassify(datToClass[:, :-1], weakClassArr)

# 计算在测试集上的准确率
test_labels = datToClass[:, -1] # 最后一列是真实标签
test_error = np.sum(predictions != test_labels) / len(test_labels)
print(f"\n 测试集上的最终错误率为: {test_error:.4f}")



















