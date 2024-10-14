import numpy as np
import random
import pandas as pd
import scipy
from matplotlib import pyplot as plt

velos = np.loadtxt("icevelocity.txt")  # load in dataset

z = velos[:,0]  # the depths of measurements (independent)
v = velos[:,1]  # velocity at given depth (dependent)



def rmse(dataset, model):
    """
    function for calculating rmse
    :param dataset: 2D dataset with independent variable in first column, dependent in second column
    :param model: a lamda function that has approximates dataset
    :return: rmse
    """
    sum = 0
    for i in range(len(dataset)):
        sum += (model(dataset[i][0]) - dataset[i][1]) ** 2
    radicand = sum / len(dataset)
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

    return train_set, test_set, train_index, test_index

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

def cross_val(dataset, percent, degree):
    '''
    partitions dataset into two subsets, trains a model on one subset and tests (calculates rmse) on other
    :param dataset: 2D array, input in first column, output in second
    :param percent: whole number percent to partition data
    :param degree: highest degree of polynomial model
    :return: root mean squared estimate
    '''
    split = getTrainTest(dataset, percent)
    train = split[0]
    test = split[1]
    train_coef = np. polyfit(train[:, 0], train[:, 1], degree)
    train_func = poly_lambda(train_coef)
    root_mean = rmse(test, train_func)
    return root_mean

def monte_rmse(trials = 1000, caca = velos, degree = 3, percent = 90):
    '''
    monte-carlo simulation for cross validation
    :param trials: # of trials
    :param caca: 2D dataset
    :param degree: highest degree of parametric polynomial model
    :param percent: whole number percent to partition dataset
    :return: 1 x tirals rmse for each trial
    '''
    deg_rmse = np.zeros((trials, 1))
    for i in range(0, trials):
        deg_rmse[i] = cross_val(caca, percent, degree)
    return deg_rmse


def mov_avg(x, y, w_size, weight=False):
    '''
    calculates moving window average of a dataset
    :param x: independent variable of a dataset
    :param y: dependent variable of a dataset
    :param w_size: window size
    :param weight: default false. whether to calculate weighted moving window average
    :return: moving average of y-values
    '''
    mov_mean = np.zeros(len(x))  # initiate array
    if not weight:
        count = 0
        for i in x:
            xlow = i - w_size / 2
            xhigh = i + w_size / 2
            Ix = np.logical_and(x > xlow, x < xhigh)
            wind_avg = np.mean(y[Ix])
            mov_mean[count] = wind_avg
            count +=1
    else:
        count = 0
        for i in x:
            xlow = i - w_size / 2
            xhigh = i + w_size / 2
            Ix = np.logical_and(x > xlow, x < xhigh)
            y_mod = y[Ix]
            weights = []
            for j in x[Ix]:
                weight = (15 / 16) * (1 - ((j - i) / (w_size / 2)) ** 2) ** 2
                weights.append(weight)
            ymod_denom = sum(weights)
            ymod_num = y_mod * weights
            ymod = sum(ymod_num) / ymod_denom
            mov_mean[count] = ymod
            count += 1
    return mov_mean


####################################################################################

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

'''
#plot ice velocities and polynomial models w/ degrees 0-4
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
'''


monte0 = monte_carlo_param(velos, 0, 90)
monte1 = monte_carlo_param(velos, 1, 90)
monte2 = monte_carlo_param(velos, 2, 90)
monte3 = monte_carlo_param(velos, 3, 90)
monte4 = monte_carlo_param(velos, 4, 90)
'''
print(monte_stat_table(monte0, 0))
print()
print(monte_stat_table(monte1, 1))
print()
print(monte_stat_table(monte2, 2))
print()
print(monte_stat_table(monte3, 3))
print()
print(monte_stat_table(monte4, 4))
'''

deg0_rmse = monte_rmse(degree=0)
deg1_rmse = monte_rmse(degree=1)
deg2_rmse = monte_rmse(degree=2)
deg3_rmse = monte_rmse(degree=3)
deg4_rmse = monte_rmse(degree=4)
'''
# Need to come back and put finishing touches on these
plt.figure(figsize=(8,10))
plt.subplot(5,1,1)
plt.hist(deg0_rmse, 40)
plt.subplot(5,1,2)
plt.hist(deg1_rmse,40)
plt.subplot(5,1,3)
plt.hist(deg2_rmse, 40)
plt.subplot(5,1,4)
plt.hist(deg3_rmse, 40)

#  moving window averages for different window sizes
mwa3 = mov_avg(z,v, 3)
mwa10 = mov_avg(z,v,10)
mwa50 = mov_avg(z,v,50)


# need to come back and add finishing touches
plt.figure(figsize=(10, 8))
plt.plot(z, mwa3, color='blue', label='Window size = 3')
plt.plot(z, mwa10, color='green', label='Window size = 10')
plt.plot(z, mwa50, color='red', label='Window size = 50')
plt.scatter(z,v, color='black', marker='o', label='Ice Velocity')
plt.legend(loc='upper right')
'''


# weighted moving window averages for different window sizes
wmwa3 = mov_avg(z, v, 3, True)
wmwa10 = mov_avg(z, v, 10, True)
wmwa50 = mov_avg(z, v, 50, True)

'''
plt.figure(figsize=(10,8))
plt.plot(z, wmwa3, color='blue')
plt.plot(z, wmwa10, color='green')
plt.plot(z, wmwa50, color='red')
plt.scatter(z,v, color='black')
'''

def better_rmse(y_data, model):
    sum = 0
    for i in range(len(y_data)):
        sum += (model[i] - y_data[i]) ** 2
    radicand = sum / len(y_data)
    rmse_poo = np.sqrt(radicand)
    return rmse_poo

def best_rmse(y_data, model):
    return np.sqrt(np.nanmean((y_data - model) ** 2))

def wei_mov_avg(train_x, train_y, test_x, w_size):
    mov_mean = np.zeros((len(test_x)))
    for i in range(len(test_x)):
        xlow = i - w_size / 2
        xhigh = i + w_size / 2
        Ix = np.logical_and(train_x > xlow, train_x < xhigh)
        y_mod = train_y[Ix]
        x_mod = train_x[Ix]
        weights = np.zeros(len(y_mod))
        for j in range(len(x_mod)):
            weights[j] = (15 / 16) * (1 - ((x_mod[j] - test_x[i]) / (w_size / 2)) ** 2) ** 2
        mov_mean[i] = np.dot(weights, y_mod) / np.sum(weights)

    return mov_mean




all_x_train = np.zeros((82, 1000))
all_y_train = np.zeros((82, 1000))
all_x_test = np.zeros((9, 1000))
all_y_test = np.zeros((9, 1000))

for i in range(1000):
    trainset = getTrainTest(velos, 90)[0]
    testset = getTrainTest(velos, 90)[1]

    all_x_train[:, i] = (trainset[:, 0])
    all_y_train[:, i] = (trainset[:, 1])
    all_x_test[:, i] = (testset[:, 0])
    all_y_test[:, i] = (testset[:, 1])


summary_window_rmse = np.zeros((180, 2))
for w in range(0, 180):
    window_rmse = []
    for ix in range(1000):
        data_x = all_x_test[:, ix]
        data_y = all_y_test[:, ix]
        y_model = wei_mov_avg(all_x_train[:, ix], all_y_train[:, ix], data_x, w)
        window_rmse.append(best_rmse(data_y, y_model))
    summary_window_rmse[w, :] = [w, np.nanmean(window_rmse)]

for row in summary_window_rmse:
    print(row)


plt.plot(summary_window_rmse[:, 0], summary_window_rmse[:, 1], color='red')
plt.show()

'''


u_surf = v[0]

rho = 917
theta = 10
g = 9.81


A0 = np.linspace(0, 1e-6, 1000) # range of A0
exp = np.linspace(0, 4, 1000)  # range A1
phys_error = np.zeros((len(A0), len(exp))) #5 initialize rmse

for n in range(len(A0)):
    for n2 in range(len(exp)):
        phys_model = u_surf - (A0[n] * (rho * g * np.sin(np.radians(theta))) ** exp[n2]) * (z ** (exp[n2] + 1))
        phys_error[n, n2] = np.sqrt(np.mean((phys_model - v) ** 2))

minrmse = np.unravel_index(np.argmin(phys_error), phys_error.shape)

best_A = A0[minrmse[1]]
best_exp = exp[minrmse[0]]
opt_rmse = np.min(phys_error)

print(best_A, best_exp)

plt.clf()
plt.imshow(phys_error, extent=(A0.min(), A0.max(), exp.min(), exp.max()), aspect='auto',
           origin='lower', cmap='viridis', vmin=0, vmax=10)
plt.xlabel('A0')
plt.ylabel('exp')
plt.plot(best_A, best_exp, 'ro', linewidth=2, markersize=10)
plt.colorbar(label='RMSE')
plt.title('Optimal values for A0 and n in ice flow equation')
plt.show()

from scipy.optimize import minimize

def RMSEval (vars):
    A0, exp = vars
    phys_model = u_surf - (A0 * (rho * g * np.sin(np.radians(theta))) ** exp) * (z ** (exp + 1))
    phys_error = np.sqrt(np.mean((phys_model - v) ** 2))
    return phys_error

initial_A = np.arange(0, 0.000001, 0.000000001)
initial_B = np.arange(0, 2, 0.002)
result = minimize(RMSEval, [0.0])
pbest=result.x
print(pbest)


Testing normality
KS test looks for biggest separation (probability difference) between two CDF's (from datasets D1, D2)
    Outputs a p-value (probability of two datasets coming from different distributions) can't say much about coming from same dataset
    i.e. test a dataset against a normal distribution
    testing statistically significant difference
        Create a second dataset with same mean, std. and size
        np.random.normal(mu, std, size)
    2-sample KS test --> scipy.stats.ks_2samp(D1, D2) returns p value

Store moving window average at center of window
Weighted average:
    (dist/h) <-- in this case h is half of the window size
    v_mod = weights * velocities / sum(weights)
    v_mod = w_i * v_i / sum(w)
'''