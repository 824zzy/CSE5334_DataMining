import numpy as np
import matplotlib.pyplot as plt

mul1, sigma1 = [1, 0], [[0.9, 0.4], [0.4, 0.9]]
mul2, sigma2 = [0, 1.5], [[0.9, 0.4], [0.4, 0.9]]
size = 500

dummyData1 = np.random.multivariate_normal(mean=mul1, cov=sigma1, size=size)
dummyData2 = np.random.multivariate_normal(mean=mul2, cov=sigma2, size=size)

def plotAns(data, center=None):
    plt.plot(data[:, 0], data[:, 1], 'x')
    plt.plot(center[:, 0], center[:, 1], 'o')
    plt.axis('equal')
    plt.show()