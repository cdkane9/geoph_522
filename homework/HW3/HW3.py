import numpy as np
import random
import pandas as pd
import scipy
from matplotlib import pyplot as plt

def rmse(dataset, model):
    """
    function for calculating rmse
    :param dataset: 2D dataset with independent variable in first column, dependent in second column
    :param model: a lamda function that has approximates dataset
    :return: rmse
    """
    sum = 0
    for i in range(len(dataset)):
        sum += (velos[i][1] - model(velos[i][0])) ** 2
    radicand = sum / len(velos)
    rmse = np.sqrt(radicand)
    return rmse


def poly_lambda(model):
    '''
    Creates a lambda function from a given array of coefficients
    :param model: an array of polynomial coefficients
    :return: a lambda function
    '''
    return lambda x: np.polyval(model, x)


def getTrainTest(caca, pTrain):
    """
    partitions data set into training subset and testing subset
    :param caca: data set
    :param pTrain: % of data set used to train model
    :return: two arrays
    """

    pTrain /= 100  # decimal conversion
    ns = len(caca)  # number of samples
    nsTrain = int(round(pTrain * ns))  # pTrain% of samples of caca
    Ix = np.array(range(ns))  # list of indexes of caca

    train_index = np.random.choice(sorted(Ix), nsTrain, replace=False)  # nsTrain number of random samples
    train_set = caca[train_index]  # the set used to train model

    test_index = np.ones(len(Ix))  # create an array of 1's to build indices of testing set
    #  looks through indices of data used to train, and swaps 1 to 0 in corresponding indices
    for i in train_index:
        test_index[i] = 0
    test_set = caca[test_index == 1]  # builds test sets from test_index
    return train_set, test_set

def monte_carlo_param(caca, degree, percent, trials = 1000):
    '''
    monte carlo for selecting random data and fitting model to that data
    :param caca: dataset
    :param trials: number of simulation to run
    :param percent: percent of data to sample.  enter whole numbers
    :return:
    '''
    coef_df = np.zeros((trials, degree + 1))
    for i in range(trials):
        data_subset = getTrainTest(caca, percent)[0]
        coef_df[i] = np.polyfit(data_subset[:,0], data_subset[:,1], degree)
    return coef_df

def monte_stat_table(caca, degree):
    '''
    creates a pandas dataframe for the mean and standard deviation of coefficients returned from monte_carlo_param
    :param caca: return from monte_carlo_param
    :param degree: highest degree, must match number of columns from caca.  probably easy way to remove this param.  it's late and can't be bothered
    :return: summary stat table, a pd.DataFrame
    '''
    col_names = ['mean', 'std']
    summary_stats = []
    i = degree
    row_names = []
    while i >= 0:
        row_names.append(f"x^{i}")
        i -= 1
    for col in range(np.shape(caca)[1]):
        col_mean = np.mean(caca[:, col])
        col_std = np.std(caca[:, col])
        summary_stats.append([col_mean, col_std])

    summary_stats = pd.DataFrame(summary_stats, columns=col_names, index=row_names)
    return summary_stats


####################################################################################
velos = np.loadtxt("icevelocity.txt")  # load in dataset

z = velos[:,0]  # the depths of measurements (independent)
v = velos[:,1]  # velocity at given depth (dependent)


# calculate coefficients of models with degrees 0-4
fit0 = np.polyfit(z,v,0)
fit1 = np.polyfit(z,v,1)
fit2 = np.polyfit(z,v,2)
fit3 = np.polyfit(z,v,3)
fit4 = np.polyfit(z,v,4)


#create lambda functions for each model to calculate rmse

f0 = poly_lambda(fit0)
f1 = poly_lambda(fit1)
f2 = poly_lambda(fit2)
f3 = poly_lambda(fit3)
f4 = poly_lambda(fit4)


plt.scatter(z, v, color='black', linewidths=0.75, marker='x')  # plot velocity data
x = np.linspace(0,180)  # define domain for plotting models and creating lambda functions

plt.hlines(fit0, 0, 180, color='red', label=round(rmse(velos, f0).item(),2))

y1 = fit1[1] + fit1[0] * x
plt.plot(x, y1, color="blue", label=round(rmse(velos, f1), 2))

y2 = fit2[2] + fit2[1] * x + fit2[0] * x ** 2
plt.plot(x, y2, color="green", label=round(rmse(velos, f2), 2))

y3 = fit3[3] + fit3[2] * x + fit3[1] * x **2 + fit3[0] * x ** 3
plt.plot(x, y3, color="yellow", label=round(rmse(velos, f3), 2))

y4 = fit4[4] + fit4[3] * x + fit4[2] * x **2 + fit4[1] * x **3 + fit4[0] * x ** 4
plt.plot(x, y4, color="purple", label=round(rmse(velos, f4), 2))

plt.title('Polynomial Models for Ice Velocity', loc='center')
plt.legend(title="RMSE", loc="lower left")
#plt.show()

row_names = ['mean', 'std']


monte0 = monte_carlo_param(velos, 0, 90)
monte1 = monte_carlo_param(velos, 1, 90)
monte2 = monte_carlo_param(velos, 2, 90)
monte3 = monte_carlo_param(velos, 3, 90)
monte4 = monte_carlo_param(velos, 4, 90)

print(monte_stat_table(monte0, 0))
print()
print(monte_stat_table(monte1, 1))
print()
print(monte_stat_table(monte2, 2))
print()
print(monte_stat_table(monte3, 3))
print()
print(monte_stat_table(monte4, 4))