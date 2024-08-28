import numpy as np
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

bar1 = plt.boxplot(depths1)
plt.show()

