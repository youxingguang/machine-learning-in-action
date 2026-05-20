import numpy as np
import matplotlib.pyplot as plt
#与简单版SVM 的区别在于 SMO在alpha_i,alpha_j的选择上

"""
简化版SMO  alpha_i是循环遍历得到,alpha_j 是随机选择的，依赖概率性，在大规模数据会比较慢

    希望获得一些启发式方法  
           (1) 用外循环选择第一个alpha, 有两个方法：在所有数据进行单遍扫描，另一种则是非边界的alpha 选择
                                     非边界alpha 是 不等于 0或C 的 
           (2) 然后通过内循环选择 第二个alpha  「最大步长」 
                 在简化版，选择j后计算错误率E_j, 枚举计算, 选择使 E_i-E_j最大的
                                                                    
"""

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=np.array(dataMatIn)
        self.labelMat=np.array(classLabels).reshape(-1, 1)
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMatIn)[0]
        self.alphas=np.zeros((self.m,1))
        self.b=0
        self.eCache=np.zeros((self.m,2))

   #计算误差 E_k=f(x_k)-y_k
def calcEk(oS,k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T @(oS.X @ oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

    #选择 j
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = np.array([1, Ei]) #设定 选择i 标记1
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0] #将eCache的第0列的非0 索引取出
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            # 最大步长
            deltaE = abs(Ei - Ek)
            # maxK 记录最大步长的索引
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 如果不存在非0 索引，则随机选择
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# 选择第二个 alpha
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.labelMat[i]*Ei<oS.C)) or \
            ((oS.labelMat[i]*Ei>oS.tol) and (oS.labelMat[i]*Ei>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if(L==H):
            print("L==H")
            return 0
        eta=2.0*oS.X[i,:]@oS.X[j,:].T-oS.X[i,:]@oS.X[i,:].T-\
            oS.X[j,:]@oS.X[j,:].T
        if eta>=0:
            print("eta>=0")
            return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<1e-5):
           return 0 # j的修改量不足
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*\
                      (alphaJold-oS.alphas[j])
        updateEk(oS,i)
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
           oS.X[i,:]@oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
           oS.X[i,:]@oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
           oS.X[i,:]@oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
           oS.X[j,:]@oS.X[j,:].T
        if(0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
            oS.b=b1
        elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

#kTup 是kernel tuple 为核函数预留 'lin'-线性核, 线性核不需要设置参数，设0
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS=optStruct(np.array(dataMatIn),np.array(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged=0
        if entireSet:
            #遍历所有值
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print(f'遍历全集 {iter}:{i} pairs changed {(iter,i,alphaPairsChanged)}')
            iter+=1
        else:
            # 遍历非边界值
            nonBoundIs=np.nonzero((oS.alphas>0)*(oS.alphas<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print(f'非边界{iter}:{i} pairs changed {(iter,i,alphaPairsChanged)}')
            iter+=1
        #遍历全集 和 遍历非边界值 交替进行
        if entireSet:
            entireSet=False # 如果遍历一次全集后，禁
        elif alphaPairsChanged==0:
            entireSet=True
        print(f'迭代次数：{iter}')
    return oS.b,oS.alphas

#利用alpha 计算超平面
def calcWs(alphas,dataArr,classLabels):
    X=np.array(dataArr)
    labelMat=np.array(classLabels).T
    m,n=X.shape
    w=np.zeros(n)
    for i in range(m):
        w+=float(alphas[i])*labelMat[i]*X[i,:].T
    return w

def showClassifier(dataArr, labelArr, w, b, alphas,C):
    # 转 numpy
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)

    # -------- 正负样本 --------
    pos = dataArr[labelArr == 1]
    neg = dataArr[labelArr == -1]

    plt.scatter(pos[:,0], pos[:,1], s=30) #取所有行 第0列，取所有行 第1列， s-size:30 散点大小
    plt.scatter(neg[:,0], neg[:,1], s=30)

    # -------- 画超平面 --------
    w = np.array(w).flatten()

    # 取两个端点
    x1 = np.min(dataArr[:,0])
    x2 = np.max(dataArr[:,0])

    # 直线:
    y1 = (-b - w[0]*x1) / w[1]
    y2 = (-b - w[0]*x2) / w[1]

    plt.plot([x1, x2], [y1, y2])

    # 标记支持向量
    for i, alpha in enumerate(alphas):

        # alphas 可能是 matrix([[x]])
        if (alpha > 1e-5) and (alpha< C):
            x, y = dataArr[i]
            plt.scatter(x,y,s=150,facecolors='none',edgecolors='red',linewidths=1.5)
    plt.show()

def loadDataSet(fileName):
    dataMat=[];labelMat=[] #新建两个空列表
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

fileName=r"E:\机器学习笔记\SVM\testSet.txt"
dataArr,labelArr=loadDataSet(fileName)
C=0.6
toler=0.001
maxIter=40
b,alphas=smoP(dataArr,labelArr,C,toler,maxIter)
w=calcWs(alphas,dataArr,labelArr)
showClassifier(dataArr, labelArr, w, b, alphas,C)












