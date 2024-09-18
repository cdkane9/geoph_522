import pandas as pd
import numpy as np
import random
import scipy as sp
from scipy.integrate import quad
from scipy import stats as st
import matplotlib.pyplot as plt

elevations_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/elevations.txt"

elevations = np.loadtxt(elevations_path)


#remove nan's
elevations = elevations[np.logical_not(np.isnan(elevations))]

#1D version of elevations
elev = elevations.ravel()


#calculate the max, min, mean, and standard deviation of elevations dataset
min = np.nanmin(elev)
max = np.max(elev)
mu = np.mean(elev)
std = np.std(elev, ddof = 1)


def rdh(caca, num_bins = 10):
    # define a function that will output rdh.  easier this way as will only
    #   write code once.  'dataset' and 'width' passed as arguments to make it easier to
    #   play around with histogram binwidth or individual datasets

    #np.histogram returns two arrays, one for nc (number of counts) and one for xvals of bins
    counts, xbins = np.histogram(caca, bins = num_bins)

    #calculate the difference between bin limits
    dx = xbins[1] - xbins[0]

    #calculate the area of each bin
    area = counts * dx

    #divides each count by the sum total area of histogram
    #outputs array of normalized bin heights
    rdh = counts / sum(area)

    # plt.bar(x coords of bars, heights, bin_width)
    # xbins [:-1] + dx / 2 sets the x coordinates to be in the middle of the bin
    #   i.e. so if the bins are 10-20, 20-30, 30-40, dx = 10, start at first xbin and add 5 (dx/2)
    #   then xcoords will be 15, 25, 35
    rdh = plt.bar(xbins[:-1] + dx / 2, rdh, width=dx, edgecolor='black')

    return rdh


def pdf(caca, xll, xul, yll, yul):
    #############################
    #create probability density function
    #inputs: 1D dataset, lower/upper limits for x and y
    #outputs: plot of probability density function

    # calculate mean for dataset
    mu = np.mean(caca)

    #calcuate std.  ddof ==> delta degrees of freedom.  unsure
    std = np.std(caca, ddof = 1)

    #np.linspace returns evenly space numbers over interval
    #   np.linspace(start, stop, number of samples to generate)
    #   this will define domain for pdf
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)

    #breaking up normal distribution function into three components

    #a is coefficient
    a = 1 / (std * np.sqrt(2 * np.pi))

    #b is numerator of exponent
    b = -(x - mu) ** 2

    #c is denominator of exponent
    c = 2 * (std ** 2)

    #putting together a,b,c for pdf
    f = a * np.exp(b/c)

    return plt.plot(x, f, 'r', linewidth = 3), plt.xlim(xll, xul), plt.ylim(yll, yul)

def CDF(caca, llim = 0, ulim = np.inf):
    #function for calculating cumulative probability
    #inputs: dataset, lower limit, upper limit
        # default args for ease later on, also to check CDF(caca) = 1
    #outputs: definite integral value, or cumulivitve probability between lower and upper limit
    #calculate pdf
    mu = np.mean(caca)
    std = np.std(caca, ddof = 1)
    a = 1 / (std * np.sqrt(2 * np.pi))
    c = 2 * std ** 2

    #creates a function, essential f(x) for pdf with domain x
    pdf = lambda x: a * np.exp((-(x - mu) ** 2) / c)

    #calculates definite integral; quad(function, lower bound, upper bound)
    CDF = quad(pdf, llim, ulim)

    #quad returns evaluation and uncertainty.  CDF func will return only the first value
    return CDF[0]


def monte_carlo(caca, trials = 1000, samples = 10):
    # monte carlo simulation
    # inputs: dataset, #of trials, #of samples from each trial
    # outpus: 4 lists, each element is specified value from each trial
    # For each trial, 10 data points are taken.  the mean, min, max, and std are
    #   taken for each trial, and added to the associated list

    mean_vals = []
    min_vals = []
    max_vals = []
    std_vals = []
    for i in range(trials):
        rand = random.sample(sorted(caca), samples)
        mean_vals.append(np.mean(rand))
        min_vals.append(np.min(rand))
        max_vals.append(np.max(rand))
        std_vals.append(np.std(rand))
    return mean_vals, min_vals, max_vals, std_vals




#creates vars so don't have to call function each time
#1000 trials, 10 data points each trial
mean_vals1 = monte_carlo(elevations)[0]
min_vals1 = monte_carlo(elevations)[1]
max_vals1 = monte_carlo(elevations)[2]
std_vals1 = monte_carlo(elevations)[3]
"""
#create 1 plot that has rdh and pdf for all elevations mean elevations (from monte carlo), max, min, std. dev
#plt.figure(figsize = (6,6))
plt.subplot(5,1,1)
pdf(elev, 2725, 2925, 0, 0.02)
rdh(elev, 50)
plt.annotate("Elevations", xy = (2850, 0.01))

plt.subplot(5,1,2)
rdh(mean_vals1, 30)
pdf(mean_vals1, 2725, 2925, 0, 0.045)
plt.annotate("Mean Elevations", xy = (2850, 0.02))

plt.subplot(5,1,3)
rdh(max_vals1, 45)
pdf(max_vals1, 2725, 2925, 0, 0.03)
plt.annotate("Max Elevations", xy = (2775, 0.015))

plt.subplot(5,1,4)
rdh(min_vals1, 30)
pdf(min_vals1, 2725, 2925, 0, 0.055)
plt.annotate("Min Elevations", xy = (2800, 0.02))

plt.subplot(5,1,5)
rdh(std_vals1,40)
pdf(std_vals1, 0, xul = 70, yll = 0, yul = 0.055)
plt.annotate("Std. Deviations", xy = (10, 0.04))

plt.show()
"""

def vary_samp_size(start, stop, step, dataset, n_trials):
    """
    Function for varying the sample size for Monte Carlo Simulation
    Inputs:
        "start": int, smallest number of samples
        "stop": int, largest number of samples
        "step": int, self-explanatory.  See range() below
        "dataset": 1 or 2D array
        "n_trials": how many times to repeat random sampling
        "stat": statistic to measure for each trial.  Ended up being easier to just take one stat at a time, and call function 4 times
            Now that I think about it, will need to go back and fix function so that with one call, it returns 4 data frames,
            One for each statistic
    Outputs:
        2D array.  For given 'stat', let's say mean, columns will be mean_mean, mean_min, mean_max, mean_std.
    """
    samp_size = range(start, stop, step)
    result_of_data = []
    colnames = ["samples", "mean_mean", "std_mean", "mean_min", "std_min", "mean_max", "std_max", "mean_std", "std_std"]
    for i in range(len(samp_size)):
        #temp_result is a place holder for the monte carlo simulation
        temp_result = monte_carlo(caca=dataset, samples=samp_size[i], trials=n_trials)
        row = samp_size[i]
        result_of_data.append([
            row,
            np.mean(temp_result[0]),
            np.std(temp_result[0]),
            np.mean(temp_result[1]),
            np.std(temp_result[1]),
            np.mean(temp_result[2]),
            np.std(temp_result[2]),
            np.mean(temp_result[3]),
            np.std(temp_result[3])

        ])
    final = pd.DataFrame(result_of_data, columns=colnames)
    return final


varied_monte_carlo = vary_samp_size(10, 100, 10, elevations, 1000)
print(varied_monte_carlo)


def scatterer(stat):
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"], label="Mean", color="navy", s=25, marker='.')
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"] + varied_monte_carlo[f"std_{stat}"], label=f"{stat}", color="red", s=25, marker=".")
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"] - varied_monte_carlo[f"std_{stat}"], label=f"{stat}", color="green", s=25, marker=".")

plt.subplot(1,4,1)
scatterer("mean")

plt.subplot(1,4,2)
scatterer("min")

plt.subplot(1,4,3)
scatterer("max")

plt.subplot(1,4,4)
scatterer("std")
# Show plot

plt.show()






unc_min = round(CDF(elevations, 0.995 * np.min(elevations), 1.005 * np.min(elevations)), 3) * 100
unc_max = round(CDF(elevations, 0.995 * np.max(elevations), 1.005 * np.max(elevations)), 3) * 100





#for questions 8, 9: what is probability that an avg value measured from MC simulation is less than true mean (of entire elevations dataset)
print(
    f"Question 8:\n",
    f"The probability of measuring a value less than the true mean is "
    f"{round(CDF(elevations, ulim = np.mean(elevations)), 2) * 100}% \n")
print(
    f"Question 9: \n",
    f"The probability of measuring a value within 1% of the minimum is {unc_min}% \n",
    f"The probability of measuring a value within 1% of the maximum is  {unc_max}% \n"
)

print(
    f"Question 10: \n",
    f"68% of the the measured mean elevations fall between {round(np.mean(mean_vals1) - np.std(mean_vals1), 2)}m "
    f"and {round(np.mean(mean_vals1) + np.std(mean_vals1), 2)}m"
)
