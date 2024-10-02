import numpy as np
import warnings
import pandas as pd
import scipy
from matplotlib import pyplot as plt

velos = np.loadtxt("icevelocity.txt")


z = velos[:,0]

v = velos[:,1]

plt.scatter(z, v)


fit0 = np.polyfit(z,v,0)
fit1 = np.polyfit(z,v,1)
fit2 = np.polyfit(z,v,2)
fit3 = np.polyfit(z,v,3)
fit4 = np.polyfit(z,v,4)
fit5 = np.polyfit(z,v,5)

x = np.linspace(z[0], z[-1])

y0 = fit0
plt.hlines(fit0, z[0], z[-1])

y1 = fit1[1] + fit1[0] * x
plt.plot(x, y1, color="orange")

y2 = fit2[2] + fit2[1] * x + fit2[0] * x **2
plt.plot(x, y2, color="red")

y3 = fit3[3] + fit3[2] * x + fit3[1] * x **2 + fit3[0] * x **3
plt.plot(x, y3, color="green")

y4 = fit4[4] + fit4[3] * x + fit4[2] * x **2 + fit4[1] * x **3 + fit4[0] * x **4
plt.plot(x, y4, color="purple")

y5 = fit5[5] + fit5[4] * x + fit5[3] * x **2 + fit5[2] * x **3 + fit5[1] * x **4 + fit5[0] * x **5
plt.plot(x, y5, color="pink")



f1 = lambda x: fit1[1] + fit1[0] * x

f2 = lambda x: fit2[2] + fit2[1] * x + fit2[0] * x **2

f3 = lambda x: fit3[3] + fit3[2] * x + fit3[1] * x **2 + fit3[0] * x **3

f4 = lambda x: fit4[4] + fit4[3] * x + fit4[2] * x **2 + fit4[1] * x **3 + fit4[0] * x **4

f5 = lambda x: fit5[5] + fit5[4] * x + fit5[3] * x **2 + fit5[2] * x **3 + fit5[1] * x **4 + fit5[0] * x **5



def rmse(actual, predicted):
    """
    #CDK 20240927 1444MST
    #Calculates root mean squared error
    #:param actual:
    #:param predicted:
    #:return:
    """
    radicand = 0
    for depth in actual[0]:
       radicand += (actual[1] - predicted(depth)) ** 2
    radicand /= len(actual)
    rmse = np.sqrt(radicand)
    return rmse


def getTrainTest(caca, pTrain):
    """
    Split a dataset 90/10, test on the 90% and test on the remaining 10%
    :param caca: 2D dataset
    :return: training dataset, test dataset
    """
    pTrain /= 100
    ns = len(caca)
    nsTrain = int(round(pTrain * ns))
    Ix = np.array(range(ns))
    train_index = np.random.choice(sorted(Ix), nsTrain, replace=False)
    test_index = np.ones(len(Ix))
    train_set = velos[train_index]
    test_set = []
    count = 0
    for i in test_index:
        count += 0
        for i in test_index:
            if i == 1:
                test_set.append(velos[count])
            count += 1

    return train_set

print(getTrainTest(velos, 90))

