import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy import stats as st
import matplotlib.pyplot as plt


#paths for depth data
depths1_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/GMdepth1.txt"
depths2_path = "/Users/colemandavidkane/Documents/BSU/GEOPH_522/homework/GMdepth2.txt"

#import depth data into nparrays
depths1 = np.loadtxt(depths1_path)
depths2 = np.loadtxt(depths2_path)


#function for calcuating iqr by finding difference between q3 and q1 for a dataset
def iqr_calc(caca):
    #np.percentile(dataset, quantile)
    q1 = np.percentile(caca,25)
    q3 = np.percentile(caca, 75)
    iqr = q3 - q1
    return iqr

#func for finding mean, std, median, mode, iqr, skewness, and kurtosis
#func adds two lists together into pandas data frame
def stat_table(caca):
    stats = ["mean", "std", "median", "mode", "iqr", "skewness", "kurtosis"]
    vals = [
        np.mean(caca),
        np.std(caca, ddof = 1),
        np.median(caca),
        st.mode(caca, axis = None)[0],
        iqr_calc(caca),
        st.skew(caca),
        st.kurtosis(caca)
    ]
    df = pd.DataFrame({"stats": stats, "values": vals})
    return df


#prints out the two stat tables w/ crude title
print(f"GMdepth1 \n {stat_table(depths1)}")
print()
print(f"GMdepth2 \n {stat_table(depths2)}")
print()


######################################################################
#The following code creates a histogram and boxplot for each depth transect (depths1 and depths2)


#establishes figure with all four plots to be made
#Lindsay helped me with arranging all 4 plots in one figure
#pt.subplot(nrows, ncols, location)  [[1,2],[3,4]]
plt.subplot(2,2,1)

#create box and whisker plot for depths1
box1 = plt.boxplot(depths1, vert = False)

#set plot limits and add title
plt.xlim(50,140)
plt.title("Transect 1")

#######

#dictate location of second boxplot
#the following are the same commands as above
plt.subplot(2,2,2)
box2 = plt.boxplot(depths2, vert = False)
plt.xlim(50,140)
plt.title("Transect 2")

#######

plt.subplot(2,2,3)
#create histogram for first transect with 30 bins
hist1 = plt.hist(depths1, 30)

#plot limits
plt.xlim(50,140)
plt.ylim(0,6)

#######

#same commands for above histogram
plt.subplot(2,2,4)
hist2 = plt.hist(depths2, 30)
plt.xlim(50,140)
plt.ylim(0,6)

plt.show()

######################################################################


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

    return plt.plot(x, f, 'r', linewidth = 3), plt.xlim([.9 * min(x), 1.1 * max(x)]), plt.ylim([0,0.04])


#arrange plots for rdh and pdf for both sites into one figure
plt.subplot(1,2,1)
rdh(depths1)
pdf(depths1)

plt.subplot(1,2,2)
rdh(depths2, 12)
pdf(depths2)
plt.tight_layout()

plt.show()



def CDF(caca, llim = 0, ulim = np.inf):
    #function for calculating cumulative probability
    #inputs: dataset, lower limit, upper limit
        # default args for ease later on, also to check CDF(caca) = 1
    #outputs: definite integral value, or cumulivitve probability between lower and upper limit
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



#evaluates CDF 20cm above and below the mean for question 9
q9_1 = round(CDF(depths1, np.mean(depths1) - 20, np.mean(depths1) + 20), 3) * 100

q9_2 = round(CDF(depths2, np.mean(depths2) - 20, np.mean(depths2) + 20), 3) * 100



#answering questions at bottom of HW document
print(
    "Question #9 \n",
      "What is the probability of a new measurement at each site being with 20cm of the average value? \n",
      f"For site 1:  {q9_1}%\n",
      f"For site 2: {q9_2}%"
    )
print()
print(
    "Question #10 \n",
      "What is the probability of a new measurement at each site being at least 20cm larger than the average value?\n",
      f"For site 1: {round(CDF(depths1, np.mean(depths1) + 20), 3) * 100}%\n",
      f"For site 2: {round(CDF(depths2, np.mean(depths2) + 20), 3) * 100}%"
    )
print()
print("Question #11\n",
      "What is the probability of a new measurement at each site being at least 20cm smaller than the average value?\n",
      f"For site 1:  {round(CDF(depths1, ulim =  np.mean(depths1) - 20), 3) * 100}%\n",
      f"For site 2: {round(CDF(depths2, ulim =  np.mean(depths2) - 20), 3) * 100}%"
        )







