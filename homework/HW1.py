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


plt.figure(1)
plt.boxplot(depths1, vert = False, patch_artist = True)

plt.figure(2)
#np.histogram returns two arrays, one for nc (number of counts) and one for xvals of bins (xbins)
counts, xbins =  np.histogram(depths1, bins = 15)

#calculate the difference between bin limits
dx = xbins[2] - xbins[1]

#calculate the area of each bin, sum(area) returns total area of histogram
area = counts * dx

#divides each count by the sum total area of histogram
#outputs array of nomalized bin heights
rdh = counts / sum(area)


#creates rel. density histogram
#plt.bar(x coords of bars, heights,
plt.bar(xbins[:-1] + dx / 2, rdh, width = dx, edgecolor = 'black')







"""
#creates boxplot for depths1 and depths2
box1 = plt.boxplot(depths1)
box2 = plt.boxplot(depths2)

#creates histogram for depths1 and depths2
hist1 = plt.hist(depths1, 30)
hist2 = plt.hist(depths2, 30)

plt.subplot()
plt.show()
"""

