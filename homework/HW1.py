import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt


#paths for depth data
depths1_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/GMdepth1.txt"
depths2_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/GMdepth2.txt"

#import depth data into nparrays
depths1 = np.loadtxt(depths1_path)
depths2 = np.loadtxt(depths2_path)

#calculates arithmetic mean for the two transects
mean1 = np.mean(depths1)
mean2 = np.mean(depths2)

#calculates median for the two transects
med1 = np.median(depths1)
med2 = np.median(depths2)

#calculates mode for two transects
#st.mode returns value and count, '[0]' added to just get value
mode1 = st.mode(depths1)[0]
mode2 = st.mode(depths2)[0]

#func for calculating iqr by finding difference between q3 and q1 for a dataset
def iqr_calc(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return iqr

#assigns value returned from iqr_calc to variables
iqr1 = iqr_calc(depths1)
iqr2 = iqr_calc(depths2)

#calculates skewness for depths 1 and depths 2
skew1 = st.skew(depths1)
skew2 = st.skew(depths2)

#calculates kurtosis for depths1 and depths2
kurt1 = st.kurtosis(depths1)
kurt2 = st.kurtosis(depths2)

#create box and whisker plot for depths1
plt.figure(1)
plt.boxplot(depths1, vert = False, patch_artist = True)

plt.figure(2)
#np.histogram returns two arrays, one for nc (number of counts) and one for xvals of bins (xbins)
counts, xbins =  np.histogram(depths1, bins = 10)

#calculate the difference between bin limits
dx = xbins[2] - xbins[1]

#calculate the area of each bin, sum(area) returns total area of histogram
area = counts * dx

#divides each count by the sum total area of histogram
#outputs array of nomalized bin heights
rdh = counts / sum(area)


#creates rel. density histogram
#plt.bar(x coords of bars, heights, bin width)
#xbins[:-1] + dx / 2 sets the x coordinate to be in the middle of the bin
    #i.e. so if the bins are 10-20, 20-30, 30-40, dx = 10, start at first xbin and add 5 (dx / 2)
    # so that xcoords will be 15, 25, 35.
plt.bar(xbins[:-1] + dx / 2, rdh, width = dx, edgecolor = 'black')



########################################################
#create probability density function

#calculate std.  ddof ==> delta degrees of freedom.  unsure
std = np.std(depths1, ddof = 1)

#find mean
mu = np.mean(depths1)

#np.linspace returns evenly spaced numbers over interval
    #np.linspae(start, stop, number of samples to generate)
    #this defines the domain for pdf
x = np.linspace(mu - 3 * std, mu + 3 * std, 100)

#breaking up normal distribution function into three components

#a is coefficient
a = 1 / (2 * std)

#b is numerator of exponent
b = -(x - mu) ** 2

#c is denominator of exponent
c = 2 * std

f = a * np.exp(b / c)

plt.plot(x, f, 'r', linewidth = 3)
plt.xlim([mu - 3 * std, mu + 3 * std])
plt.ylim([0, 1.1 * max(f)])
plt.show()








