import matplotlib.pyplot as plt
import pandas as pd
import random



dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10

selectedAds = []

numberOfRewards0 = [0]*d
numberOfRewards1 = [0]*d
totalReward = 0
for n in range(0, N):
    ad = 0
    maxRandom = 0
    for i in range(0, d):

        randomBeta = random.betavariate(numberOfRewards1[i] + 1, numberOfRewards0[i] + 1)

        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i

        selectedAds.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numberOfRewards1[i] += 1
        elif reward == 0:
            numberOfRewards0[i] += 1

        totalReward += reward


plt.hist(selectedAds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

print(totalReward)