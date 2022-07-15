import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf





x_str = 'Ehsan157'
y_str = 'Poormohammady'


for i, (x_ch, y_ch) in enumerate(zip(x_str[1:], y_str[1:]), 0):
    print('{} : {}-{}'.format(i, x_ch, y_ch))



# Creates 50 random x and y numbers
np.random.seed(1)
n = 50
x = np.random.randn(n)
y = x * np.random.randn(n)

# Makes the dots colorful
colors = np.random.rand(n)

# Plots best-fit line via polyfit
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

# Plots the random x and y data points we created
# Interestingly, alpha makes it more aesthetically pleasing
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()