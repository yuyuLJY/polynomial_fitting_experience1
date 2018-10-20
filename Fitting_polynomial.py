# coding=utf-8
# 在0-2*pi的区间上生成100个点作为输入数据
import numpy
from matplotlib import pyplot as pt

# 对输入数据加入gauss噪声
# 定义gauss噪声的均值和方差
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

def draw_picture(X,Y,p):
    # 画出这些点
    pt.plot(X, Y, linestyle='', marker='.')
    pt.plot(X,p(X))
    pt.show()

def polyfit(x, y, degree):
    results = {}
    coeffs = numpy.polyfit(x, y, degree)
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    print('多项式：', p)
    return p

X,Y = creat_data(100)
Xtest,Ytest = creat_data(50)
print(X)
print(Y)
for i in range(18):
    p = polyfit(X, Y, i)
    loss =0
    for j in range(len(Xtest)):
        # print('预测yi 和真实 y', p(Xtest[i]), Ytest[i])
        loss = loss + (p(Xtest[j])-Ytest[j])**2
    print('loss', i, ' :', loss)
    draw_picture(Xtest, Ytest, p)

print("ending")