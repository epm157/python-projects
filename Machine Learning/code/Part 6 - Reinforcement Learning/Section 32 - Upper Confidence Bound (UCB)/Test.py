import matplotlib.pyplot as plt
import pandas as pd
import random
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10

'''

adsSelected = []
totalReward = 0

for n in range(0, N):
    ad = random.randrange(d)
    adsSelected.append(ad)
    reward = dataset.values[n, ad]
    totalReward += reward

plt.hist(adsSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

'''


selectedAds = []
numberOfSelections = [0]*d
sumOfRewards = [0]*d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxUpperBound = 0
    for i in range(0, d):
        if numberOfSelections[i] == 0:
            upperBound = 1e400
        else:
            averageReward = sumOfRewards[i] / numberOfSelections[i]
            delta_i = math.sqrt(3/2 * math.log(i+1))/numberOfSelections[i]
            upperBound = averageReward + delta_i
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
        selectedAds.append(ad)
        numberOfSelections[ad] += 1
        reward = dataset.values[n, ad]
        sumOfRewards[ad] += reward
        totalReward += reward


plt.hist(selectedAds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

