import numpy as np
from data import *
from scipy.spatial.distance import cdist

class K_means():
    def __init__(self, X, k, c):
        self.X = X
        self.k = k
        self.c = c
        self.i = 0
    
    def __call__(self, max_iter=10000):
        diag = np.eye(self.k)
        self.c = self.X[np.random.choice(len(self.X), self.k, replace=False)]
        for i in range(max_iter):
            self.i = i
            prev_c = np.copy(self.c)
            dist = cdist(self.X, self.c)
            cluster_idx = np.argmin(dist, axis=1)
            cluster_idx = diag[cluster_idx]
            self.c = np.sum(self.X[:, None, :] * cluster_idx[:, :, None], axis=0) / np.sum(cluster_idx, axis=0)[:, None]
            if np.allclose(prev_c, self.c, atol=1e-3):
                break
            
            
                
            
            
if __name__ == "__main__":
    k_clusters = 4
    centerR = dummyData1[np.random.choice(len(dummyData1), k_clusters, replace=False)]
    center1 = np.array([[10., 10.], [-10., -10.]])
    center2 = np.array([[10., 10.], [-10., -10.], [10., -10.], [-10., 10.]])
    
    k1 = K_means(X=dummyData1, k=2, c=center1)
    k1.__call__()
    print("The center of dummy data 1 is {}.".format(k1.c))
    print("After {} iteration.".format(k1.i))
    
    k2 = K_means(X=dummyData2, k=4, c=center2)
    k2.__call__()
    print("The center of dummy data 2 is {}.".format(k2.c))
    print("After {} iteration.".format(k2.i))
    
    plotAns(dummyData1, k1.c)
    plotAns(dummyData2, k2.c)
    