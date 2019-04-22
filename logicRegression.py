import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

path = 'E:/梯度下降/'+'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path,header=None,names=["Exam 1","Exam 2",'Admitted'])
#因为第一行不是列名而是数据，所以把header设置为None
#测试数据是否能够读取
#print(pdDate.head(10))
#print(pdDate.shape)
positive = pdData[pdData['Admitted'] == 1]# Admintted 为1时表示该成绩被录取
negative = pdData[pdData['Admitted'] == 0]#Admintted 为0时表示该成绩未被录取

#展现初始数据图像
# fig,ax = plt.subplots(figsize = (10,5))
# ax.scatter( positive['Exam 1'], positive['Exam 2'], s = 30, c = 'b', marker = 'o', label = 'Admintted')
# ax.scatter( negative['Exam 1'], negative['Exam 2'], s = 30, c = 'r', marker = "x", label = 'Not Admintted')
# ax.legend()#左上角标签
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# ax.set_title('inital data')
# plt.show()

#设置sigmoid 函数 𝑔(𝑧)=1/(1+𝑒^(−𝑧))
#
def  sigmoid(z):
    '''
    sigmoid 函数 𝑔(𝑧)=1/(1+𝑒^(−𝑧))
    ℝ→[0,1] z→(−∞,+∞)
    𝑔(0)=0.5
    𝑔(−∞)=0
    𝑔(+∞)=1
    :param z:输入的数字
    :return: z所对应的概率
    '''
    return 1 / (1 + np.exp(-z))
#测试sigmoid函数
# nums = np.arange(-10,10,step = 0.1)
# fig,ax = plt.subplots(figsize=(10,5))
# ax.plot(nums,sigmoid(nums),'r')
# ax.set_title('sigmoid function')
# plt.show()

#计算X与θ组合后的概率
def model(X,theta):
    '''
    传递两个向量（X 和θ）进来，进行点乘后传入sigmoid函数中 即
    ℎ𝜃(𝑥)=(𝜃0,𝜃1,𝜃2)✖(1,x1,x2)T = 𝜃0+ 𝜃1x1 + 𝜃2x2
    :param X:输入数据
    :param theta:数据权重
    :return:在输入该权重下数据后的概率
    '''
    return sigmoid(np.dot(X,theta.T))

pdData.insert(0,'Ones',1)#添加一个𝜃0这个列所对应的X全为1  在第0列取名叫Ones 数据为1
orig_data = pdData.values #将数据的panda表示形式转换为对数组形式即一行为一个数组，100个数组组成一个大数组
#print(pdData)
#print(orig_data)
cols = orig_data.shape[1]#shape[0] 是获取当前行数 shape[1] 是获取当前列数
X = orig_data[:,0:cols-1]#X为第0列到第cols-1列 即第0列至第2列 此处X（100，3）
y = orig_data[:,cols-1:cols]#Y为第3列，作为标签列使用  此处 Y(100,1)
theta = np.zeros([1,3])#创建一个1行3列的 0向量theta  此处 theta (1,3)


#平均损失函数
def cost(X, y, theta):
    '''
    损失函数 损失函数是所有模型误差的平方和
    D(hθ(x),y)= −ylog(hθ(x))−(1−y)log(1−hθ(x))
    平均损失函数 J(θ)
    J(θ)=1/𝑛∑𝑖=1 𝑛 𝐷(ℎ𝜃(𝑥𝑖),𝑦𝑖)
    :param X: 输入数据
    :param y: 数据标签
    :param theta: 数据权重
    :return: 平均损失函数 J(θ)
    '''
    left = np.multiply(-y, np.log(model(X, theta))) #−𝑦log(ℎ𝜃(𝑥))
    right = np.multiply(1 - y, np.log(1 - model(X, theta))) #(1−y)log(1−hθ(x))
    return np.sum(left-right)/(len(X)) #1/𝑛∑𝑖=1 𝑛 𝐷(ℎ𝜃(𝑥𝑖),𝑦𝑖)
#测试cost函数
#print(cost(X, y, theta))

#梯度计算
#∂J/∂θj =-1/m∑𝑖=1 m (yi−hθ(xi))xij
def gradient(X, y, theta): #计算梯度 X（100，3） Y(100,1) theta (1,3)
    '''
    :param X: 输入数据
    :param y: 数据标签
    :param theta: 数据权重
    :return: 返回当前位置梯度最大值
    '''
    grad = np.zeros(theta.shape) #对应和theta相同梯度，这里是3个（1行3列）,先全部置零
    error = (model(X, theta) - y).ravel()# 计算(ℎ𝜃(𝑥𝑖)-𝑦𝑖) =-(𝑦𝑖−ℎ𝜃(𝑥𝑖))，结果转换为一维数组 此处是计算
    for j in range(len(theta.ravel())): #循环3次
        term = np.multiply(error, X[:,j]) # 让theta0乘以X0，theta1乘以X1，theta2乘以X2 分别计算错误率
        grad[0,j] = np.sum(term)/len(X)  #求和取平均
    return grad #返回当前位置梯度最大值

#比较三种不同梯度下降方法
STOP_ITER = 0 #按照迭代次数进行停止
STOP_COST = 1 #按照损失函数几乎没啥变化停止
STOP_GRAD = 2 #按照梯度几乎没变化停止

def stopCriterion(type, value, threshold):
    """
    :param type: 停止方法
    (STOP_ITER为按照迭代次数进行停止;STOP_COST 为按照损失函数几乎没啥变化停止；STOP_GRAD 按照梯度几乎没变化停止）
    :param value:
    :param threshold:策略对应阈值
    :return:
    """
    if type == STOP_ITER:   return value > threshold
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold

#数据洗牌,增强泛化能力
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:,:cols-1]
    y = data[:,cols-1:cols]
    return X,y

#梯度下降求解
def descent(data, theta, batchSize, stopType, thresh, alpha):
    '''
    :param data: 读取的数据
    :param theta: 权重
    :param batchSize: 批量处理数据的长度
    :param stopType: 停止策略
    :param thresh: 停止策略对应阈值
    :param alpha:学习率
    :return:更新后的权重theta；循环次数，损失率costs，梯度当前grad，消耗时间
    '''
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #初始批量数据长度
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)# 计算的梯度
    costs = [cost(X, y, theta)]# 第一次的损失值
    n = len(y.ravel())
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)#小批量数据提取进行梯度计算
        k += batchSize#初始批量数据长度每次增加batchSize
        if k >= n: #当累计批量数据长度超过数据的长度后，进行重新洗牌
            k = 0
            X,y = shuffleData(data) #重新洗牌
        #θj = θj - α1/m∑𝑖=1 m (yi−hθ(xi))xij
        theta = theta - alpha*grad #更新参数
        costs.append(cost(X, y, theta))
        i += 1
        #判断是否达到跳出循环条件，达到就跳出
        if stopType == STOP_ITER:   value = i
        elif stopType == STOP_COST: value = costs
        elif stopType == STOP_GRAD: value = grad
        if stopCriterion(stopType,value,thresh): break

    return theta, i-1, costs, grad, time.time()-init_time

#梯度下降结果展示
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    '''
    :param data: 读取的数据
    :param theta: 权重
    :param batchSize: 批量处理数据的长度
    :param stopType: 停止策略
    :param thresh: 停止策略对应阈值
    :param alpha:学习率
    :return: 更新的权重
    '''
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==len(y.ravel()): strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta

#梯度下降
n=100
#使用按照迭代次数进行停止
#runExpe(orig_data, theta, n,STOP_ITER,thresh=10000,alpha=0.000001)
#Last cost: 0.63

#使用按照损失值停止
#runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
#Last cost: 0.38

#使用根据梯度变化停止
#runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05,alpha=0.001)
#Last cost: 0.49

#随机梯度下降
#runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
#Last cost: 1.47 很不稳定,再来试试把学习率调小一些

#runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000001)
#Last cost: 0.63

#runExpe(orig_data, theta, 1, STOP_COST, thresh=0.0000001, alpha=0.0000001)
#Last cost: 0.63

#小批量梯度下降
#runExpe(orig_data, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)
#Last cost: 0.66 浮动仍然较大

#对数据进行标准化 将数据按其属性(按列进行)减去其均值，然后除以其方差。
# 最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1
from sklearn import preprocessing as pp
scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
#runExpe(scaled_data, theta, n, STOP_ITER, thresh=5000, alpha=0.001)
#Last cost: 0.38
#runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002, alpha=0.0005)
#Last cost: 0.20   Duration: 169.45s
#随机梯度下降更快，但是我们需要迭代的次数也需要更多，所以还是用batch的比较合适！！
runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)
#Last cost: 0.21

#精度
#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]
scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))