import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 解决中文无法渲染问题
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def plotTree(tree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)

    firstStr = next(iter(tree))
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) / 2.0 / plotTree.totalW,
              plotTree.yOff)

    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt)

    secondDict = tree[firstStr]
    plotTree.yOff -= 1.0 / plotTree.totalD

    for key in secondDict:
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(secondDict[key],
                     (plotTree.xOff, plotTree.yOff),
                     cntrPt)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff += 1.0 / plotTree.totalD

def plotNode(nodeTxt, centerPt, parentPt):
    plt.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="<-"))

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] + cntrPt[0]) / 2.0
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    plt.text(xMid, yMid, txtString)

def getNumLeafs(tree):
    numLeafs = 0
    firstStr = next(iter(tree))
    secondDict = tree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(tree):
    maxDepth = 0
    firstStr = next(iter(tree))
    secondDict = tree[firstStr]
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth

def createPlot(tree):
    fig = plt.figure()
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    plotTree(tree, (0.5, 1.0), '')
    plt.show()