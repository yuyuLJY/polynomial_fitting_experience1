import numpy
from matplotlib import pyplot as pt

import random

def creat_data(number):
    X = numpy.linspace(0, 2 * numpy.pi,number, endpoint=True)
    Y = numpy.sin(X)
    mu = 0
    sigma = 0.03
    for i in range(X.size):
        X[i] += random.gauss(mu, sigma)
        Y[i] += random.gauss(mu, sigma)
    return X, Y

def draw_picture(X,Y,res):
    # 画出这些点
    pt.title('alpha: 0.000000000004985')
    pt.plot(X, Y, linestyle='', marker='.')
    pt.plot(X, res,linestyle='', marker='.')
    pt.show()

# 批量梯度下降法
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()                             #得到它的转置
    for i in range(0, maxIterations):
        hypothesis = numpy.dot(x, theta)
        loss = hypothesis - y
        # cost = 1/2*numpy.dot(loss, loss)
        cost = (loss * loss).sum()
        print('cost: ',cost)
        gradient = numpy.dot(xTrains, loss) / m             #对所有的样本进行求和，然后除以样本数
        theta = theta - alpha * gradient
    return theta

#预测
def predict(x, theta):
    res = numpy.dot(x, theta)
    # print('预测数据： ',res)
    return res

'''
X, Y = creat_data(5)
print('数据X： ', X)
print('数据Y： ', Y)
pt.plot(X, Y, linestyle='', marker='.')
pt.show()
'''
X = [0.05123197 ,0.09142867, 0.21980827 ,0.2982785 , 0.43512065 ,0.52119081,
 0.58604562 ,0.70894696 ,0.86077488 ,0.90750491 ,1.07582238 ,1.1846028,
 1.25776812 ,1.30845724 ,1.53336679 ,1.62832844 ,1.75038224 ,1.82229931,
 1.86052155 ,2.04226348 ,2.0867813  ,2.22279153 ,2.31904593 ,2.42074872,
 2.54157375 ,2.70466183 ,2.7169381 , 2.91388806 ,2.97480614 ,3.02679572,
 3.17169733 ,3.32234604 ,3.39701686 ,3.53458903 ,3.67757002 ,3.7324351,
 3.8843827  ,3.94511901 ,4.0715585  ,4.14857142 ,4.30114095 ,4.33098062,
 4.46524604 ,4.58453743 ,4.71722724 ,4.83979501 ,4.85379502 ,5.00630256,
 5.13836069 ,5.26563505 ,5.30255797 ,5.4577599 , 5.53832123, 5.57175478,
 5.71247922 ,5.8642948  ,5.97682209 ,6.11322411, 6.20897864 ,6.22267993]
Y = [ 0.00479144  ,0.10729618 , 0.21031123 , 0.27159533 , 0.43913545 , 0.53710455,
  0.5603982   ,0.69886113 , 0.74790177 , 0.85152446 , 0.85553602  ,0.94512635,
  0.90492679 , 0.96728261 , 0.98391643 , 1.00579324 , 0.99383325  ,0.92787679,
  0.915492   , 0.91834254 , 0.82695726  ,0.79165242 , 0.68030385  ,0.67280751,
  0.58795989  ,0.40823096 , 0.3412459   ,0.23037785 , 0.12839187  ,0.02020986,
 -0.04285814 ,-0.15797909 ,-0.27552073 ,-0.34272084 ,-0.46724283 ,-0.53733163,
 -0.68925121 ,-0.74308338 ,-0.81940919 ,-0.84879024 ,-0.91645588 ,-0.93848912,
 -1.00062766 ,-1.01715922 ,-1.00529493 ,-0.97664692 ,-0.97190975 ,-0.95386096,
 -0.88052674 ,-0.83385585 ,-0.85887512 ,-0.78741244 ,-0.71453584 ,-0.58817958,
 -0.50943442 ,-0.36316102, -0.29954306 ,-0.20939221, -0.13204249  ,0.0266904 ]
X = numpy.array(X)
Y = numpy.array(Y)
print(X)
print(Y)
listI = []
arrayData = []
for i in range(0, len(X)):
    listI.append(X[i])
    arrayData.append(list.copy(listI))
    # print(arrayData)
    del listI[0]

print('变成数组的数据',arrayData)
m, n = numpy.shape(numpy.array(arrayData))
trainData = numpy.ones((m, n+8))
trainData[:, :-1] = arrayData
trainDataArray = numpy.array(trainData)

trainDataArray[:,0] = trainDataArray[:,3]**4
trainDataArray[:,1] = trainDataArray[:,3]**3
trainDataArray[:,2] = trainDataArray[:,3]**2

# 变成平方项
print(trainDataArray)
# 生成theta
trainLabel = numpy.array(Y)
theta = numpy.ones(n+8)


alpha = 0.000004985 # 步长0.05 -0.14
maxIteration = 200000 #迭代的次数
theta = batchGradientDescent(trainDataArray, trainLabel, theta, alpha, m, maxIteration)
print('theta: ',theta)
res = predict(trainDataArray, theta)
draw_picture(X,Y,res)