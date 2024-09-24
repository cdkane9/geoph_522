import pandas as pd
import numpy as np
import random
import scipy as sp
from scipy.integrate import quad
from scipy import stats as st
import matplotlib.pyplot as plt

elevations_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/elevations.txt"

elevation = np.loadtxt(elevations_path)


#remove nan's
elevations = elevation[np.logical_not(np.isnan(elevation))]


#1D version of elevations
elev = elevations.ravel()


#calculate the max, min, mean, and standard deviation of elevations dataset
min = np.nanmin(elev)
max = np.nanmax(elev)
mu = np.nanmean(elev)
std = np.nanstd(elev, ddof = 1)


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
    rdh = counts / np.nansum(area)

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

#create 1 plot that has rdh and pdf for all elevations mean elevations (from monte carlo), max, min, std. dev
plt.figure(figsize = (6,8))
plt.subplot(5,1,1)
pdf(elev, 2725, 2925, 0, 0.02)
rdh(elev, 50)
plt.annotate("Elevations (m)", xy = (2850, 0.01))

plt.subplot(5,1,2)
rdh(mean_vals1, 30)
pdf(mean_vals1, 2725, 2925, 0, 0.045)
plt.annotate("Mean Elevations (m)", xy = (2850, 0.02))

plt.subplot(5,1,3)
rdh(max_vals1, 45)
pdf(max_vals1, 2725, 2925, 0, 0.03)
plt.annotate("Max Elevations (m)", xy = (2775, 0.015))

plt.subplot(5,1,4)
rdh(min_vals1, 30)
pdf(min_vals1, 2725, 2925, 0, 0.055)
plt.annotate("Min Elevations (m)", xy = (2800, 0.02))

plt.subplot(5,1,5)
rdh(std_vals1,40)
pdf(std_vals1, 0, xul = 70, yll = 0, yul = 0.055)
plt.annotate("Std. Deviations (m)", xy = (10, 0.04))

plt.tight_layout()
plt.show()


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


#stores varied sample size monte carlo to variable
varied_monte_carlo = vary_samp_size(10, 510, 10, elevations, 1000)


#function for plotting each of the scatter plots
def scatterer(stat):
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"], label="Mean", color="navy", s=25, marker='.')
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"] + varied_monte_carlo[f"std_{stat}"], label=f"{stat}", color="orangered", s=25, marker=".")
    plt.scatter(varied_monte_carlo["samples"], varied_monte_carlo[f"mean_{stat}"] - varied_monte_carlo[f"std_{stat}"], label=f"{stat}", color="forestgreen", s=25, marker=".")


caption = f"Values for $\mu$, min, max, and $\sigma$ calculated from Monte Carlo simulations with varied sample size"
plt.figure(figsize=(11,7))
plt.figtext(0.5, 0.005, caption, wrap=True, horizontalalignment='center')

#plot mean vals
plt.subplot(1,4,1)
scatterer("mean")
plt.axhline(mu, color="navy")
plt.title("Average Mean")
plt.xlabel("Samples")
plt.ylabel("Elevation (m)")
plt.legend([r"$\mu$", r"$\mu + \sigma$", r"$\mu - \sigma$", r"$\mu_{true}$"], loc="upper right")

#plot min vals
plt.subplot(1,4,2)
scatterer("min")
plt.axhline(min, color="navy")
plt.title("Average Minimum")
plt.xlabel("Samples")
plt.ylim(2731,2761)
plt.legend([r"$\mu$", r"$\mu + \sigma$", r"$\mu - \sigma$", r"$min_{true}$"], loc="upper right")

#plot max vals
plt.subplot(1,4,3)
scatterer("max")
plt.axhline(max, color="navy")
plt.xlabel("Samples")
plt.title("Average Maximum")
plt.ylim(2840,2924)
plt.legend([r"$\mu$", r"$\mu + \sigma$", r"$\mu - \sigma$", r"$\max_{true}$"], loc="lower right")

#plot standard deviations
plt.subplot(1,4,4)
scatterer("std")
plt.axhline(std, color="navy")
plt.xlabel("Samples")
plt.title("Average Std. Deviation")
plt.legend([r"$\mu$", r"$\mu + \sigma$", r"$\mu - \sigma$", r"$\sigma_{true}$"], loc="lower right")

# Show plot

plt.tight_layout()
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



def uniform_spacing(caca, distance, spacing):
    """
    Function for uniform sampling
    Inputs:
        caca - 2D array
        distance - the distance that the data set spans in the real world (in meters)
        spacing - the desired spacing of sampling (assumes the same in x and y direction)
    Output:
        uniform_samples - 2D array
    """
    #find the length in x and y directions for dataset
    xlen = np.shape(caca)[0]
    ylen = np.shape(caca)[1]


    #calculate the resoltuion of the data set in meters per measurement
    resolution_x = distance / xlen
    resolution_y = distance / ylen

    #determine the step, or how many values to skip to get desired spacing
    step_x = int(spacing // resolution_x)
    step_y = int(spacing // resolution_y)


    uniform_samples = caca[::step_x, ::step_y]

    return uniform_samples

#call above function for 200m spacing, then remove nan's
spacing_200 = uniform_spacing(elevation, 1000, 200)
spacing_200 = spacing_200[np.logical_not(np.isnan(spacing_200))]


#call above function for 30m spacing, then remove nan's
spacing_30 = uniform_spacing(elevation, 1000, 30)
spacing_30 = spacing_30[np.logical_not(np.isnan(spacing_30))]

#calculate basic stats
mu_200 = int(round(np.mean(spacing_200),0))
sig_200 = np.std(spacing_200)

mu_30 = int(round(np.mean(spacing_30),2))
std_30 = np.std(spacing_30)


#will plot rdh and pdf with notched boxplot above for 30m and 200m spacing
plt.figure(figsize=(12,7))

plt.subplot(2,2,3)
rdh(spacing_200, 9)
pdf(spacing_200, 2740, 2910, 0, 0.02)
plt.xlabel("Elevation (m)")

plt.subplot(2,2,4)
rdh(spacing_30, 20)
pdf(spacing_30, 2730, 2910, 0, 0.02)
plt.xlabel("Elevation (m)")

plt.subplot(2,2,1)
plt.boxplot(spacing_200, vert=False, notch=True)
plt.annotate(f"$\mu = {mu_200}$m", (np.mean(spacing_200), 1.25))
plt.title("200m Spacing")

plt.subplot(2,2,2)
plt.boxplot(spacing_30, vert=False, notch=True)
plt.annotate(f"$\mu = {mu_30}$m", (np.mean(spacing_30), 1.25))
plt.title("30m Spacing")

plt.tight_layout()
plt.show()