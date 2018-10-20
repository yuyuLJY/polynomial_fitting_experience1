import numpy
# 求出共轭复数的参数模块
def conjugate(x0, A, b):
    r0 = b - numpy.dot(A, x0)
    print('r0--shape', b.shape, A.shape,x0.shape, numpy.dot(A,x0).shape, r0.shape)
    p0 = r0
    k = 0
    r = r0
    p = p0
    x = x0
    while k >= 0:
        former_r = r  # 记录下先前的rk
        alpha = numpy.dot(numpy.transpose(r), r) / numpy.dot(numpy.dot(numpy.transpose(p), A), p)
        x = x + alpha * p
        r = r - alpha * numpy.dot(A, p)
        print('二范式r:', numpy.linalg.norm(r))
        if numpy.linalg.norm(r) <= 10 ** -6:
            break
        beta = numpy.dot(numpy.transpose(r), r) / numpy.dot(numpy.transpose(former_r), former_r)
        # print('beta', beta)
        p = r + beta * p
        k = k + 1
    return x

# 画图模块
def drawn(X,Y,result_result):
    from matplotlib import pyplot as pt
    # pt.title('λ: lnλ = 2')
    pt.plot(X, Y, linestyle='', marker='.')
    pt.plot(X, result_Y, linestyle='', marker='.')
    pt.show()

# 输入的矩阵X和结果矩阵Y
def compute_matrix(X,Y):
    listI = []
    arrayData = []
    for i in range(0, len(X)):
        listI.append(X[i])
        arrayData.append(list.copy(listI))
        # print(arrayData)
        del listI[0]

    m, n = numpy.shape(numpy.array(arrayData))  # n=1
    trainData = numpy.ones((m, n + 4))
    trainData[:, :-1] = arrayData
    trainDataArray = numpy.array(trainData)
    trainDataArray[:, 0] = trainDataArray[:, 3] ** 4
    trainDataArray[:, 1] = trainDataArray[:, 3] ** 3
    trainDataArray[:, 2] = trainDataArray[:, 3] ** 2
    trainLabel = numpy.array(Y)
    return trainDataArray,trainLabel

X = numpy.array([0.05123197 ,0.09142867, 0.21980827 ,0.2982785 , 0.43512065 ,0.52119081,
 0.58604562 ,0.70894696 ,0.86077488 ,0.90750491 ,1.07582238 ,1.1846028,
 1.25776812 ,1.30845724 ,1.53336679 ,1.62832844 ,1.75038224 ,1.82229931,
 1.86052155 ,2.04226348 ,2.0867813  ,2.22279153 ,2.31904593 ,2.42074872,
 2.54157375 ,2.70466183 ,2.7169381 , 2.91388806 ,2.97480614 ,3.02679572,
 3.17169733 ,3.32234604 ,3.39701686 ,3.53458903 ,3.67757002 ,3.7324351,
 3.8843827  ,3.94511901 ,4.0715585  ,4.14857142 ,4.30114095 ,4.33098062,
 4.46524604 ,4.58453743 ,4.71722724 ,4.83979501 ,4.85379502 ,5.00630256,
 5.13836069 ,5.26563505 ,5.30255797 ,5.4577599 , 5.53832123, 5.57175478,
 5.71247922 ,5.8642948  ,5.97682209 ,6.11322411, 6.20897864 ,6.22267993])
Y = numpy.array([ 0.00479144  ,0.10729618 , 0.21031123 , 0.27159533 , 0.43913545 , 0.53710455,
  0.5603982   ,0.69886113 , 0.74790177 , 0.85152446 , 0.85553602  ,0.94512635,
  0.90492679 , 0.96728261 , 0.98391643 , 1.00579324 , 0.99383325  ,0.92787679,
  0.915492   , 0.91834254 , 0.82695726  ,0.79165242 , 0.68030385  ,0.67280751,
  0.58795989  ,0.40823096 , 0.3412459   ,0.23037785 , 0.12839187  ,0.02020986,
 -0.04285814 ,-0.15797909 ,-0.27552073 ,-0.34272084 ,-0.46724283 ,-0.53733163,
 -0.68925121 ,-0.74308338 ,-0.81940919 ,-0.84879024 ,-0.91645588 ,-0.93848912,
 -1.00062766 ,-1.01715922 ,-1.00529493 ,-0.97664692 ,-0.97190975 ,-0.95386096,
 -0.88052674 ,-0.83385585 ,-0.85887512 ,-0.78741244 ,-0.71453584 ,-0.58817958,
 -0.50943442 ,-0.36316102, -0.29954306 ,-0.20939221, -0.13204249  ,0.0266904 ])

trainDataArray,trainLabel = compute_matrix(X,Y)  # 从数据X和Y，计算出多项式的矩阵X和Y

# 凑出A和b: Ax = b
rank, column = numpy.shape(trainDataArray)
langbuda = numpy.e ** 1
A = numpy.dot(numpy.transpose(trainDataArray), trainDataArray) + langbuda * (numpy.ones((column, column)))
b = numpy.dot(numpy.transpose(trainDataArray), trainLabel)
# x0 = numpy.ones((column, 1))
# print('x0', x0)
result_parameter = conjugate(b, A, b)  # 带入x0是有问题的！！！！！！
print('result_parameter:', result_parameter)
result_Y = numpy.dot(trainDataArray, result_parameter)  # 计算出在w参数下应该对应的Y值
drawn(X, Y, result_Y)
