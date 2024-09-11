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


min = np.nanmin(elev)
max = np.max(elev)
mu = np.mean(elev)
std = np.std(elev, ddof = 1)


def rdh(caca, num_bins = 10):
    # define a function that will output pdf and rdh.  easier this way as will only
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


def pdf(caca):
    #############################
    #create probability density function
    #inputs: 1D dataset
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
    c = 2 * std ** 2

    #putting together a,b,c for pdf
    f = a * np.exp(b/c)

    return plt.plot(x, f, 'r', linewidth = 3), plt.xlim([.9 * np.min(x), 1.1 * np.max(x)]), plt.ylim([0,0.04])

mean_vals = []
min_vals = []
max_vals = []
std_vals = []

for i in range(1000):
    rand = random.sample(sorted(elevations), 10)
    mean_vals.append(np.mean(rand))
    min_vals.append(np.min(rand))
    max_vals.append(np.max(rand))
    std_vals.append(np.max(rand))


rdh(elev, 50)


#rdh(mean_vals, 30)
#rdh(max_vals, 45)
#rdh(min_vals)
#rdh(std_vals,40
plt.show()




