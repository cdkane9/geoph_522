import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sklearn.tree
from scipy.stats import gaussian_kde
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

'''
Gini index:
    G = sum{k=1]{N}P_k(1-P_k) = 1 = sum(P_k^2)
    sum of product of 'probability of getting it right and probability of getting it wrong'
    Determines best value for choosing threshold for given input factor
Entropy:
    H = -sum{k=1}{N}P_k cdot log(k)
    Which threshold and which feature splits the data the best
'''

from scipy.io import loadmat
data = loadmat('SMPstability.mat')
X = data['X']
S = data['S'].ravel()

from sklearn import tree
Xnames = ['Xnames']
clf = DecisionTreeClassifier()
clf.fit(X,S)
plt.figure(figsize = (20,10))
tree.plot_tree(clf, feature_names = Xnames, filled=True)

#create function to evaluate different pruning levels
def evaluate_tree(cpp_alpha):
    tree = DecisionTreeClassifier(ccp_alpha=cpp_alpha)
    score = cross_val_score(tree, X, S, cv=5)
    return score.mean()

xc = np.linspace(0,0.1,25)
scores = np.zeros(len(xc))
for n in range(len(xc)):
    scores[n] = evaluate_tree(xc[n])
plt.figure(1)
plt.clf()
plt.plot(xc, scores)
plt.show()

best_level = xc[np.argmax(scores)]


clf_pruned = DecisionTreeClassifier(ccp_alpha=best_level)
clf_pruned.fit(X, S)
plt.figure(figsize=(10,20))
