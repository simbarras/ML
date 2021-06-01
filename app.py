import numpy as np

from ML.lr.lr import Lr
from ML.lr.lr_multiple import LrMultiple
from ML.lr.lr_simple import LrSimple

if __name__ == '__main__':
    # lr = LrSimple()
    # lr = LrMultiple()
    lr = Lr()

    lr.learning_rate = 0.0001
    lr.n_iteration = 1000
    lr.n_samples = 5
    lr.n_param_more = 2
    lr.verbose = True

    lr.n_features = 1

    lr.x = np.array([[1], [2], [3], [4], [5]], np.int32)
    print(lr.x.shape)
    lr.y = np.array([[4000], [4400], [5200], [6400], [8000]], np.int32)
    print(lr.y.shape)
    # lr.random_dataset(10)

    lr.run("Grad")  # Run the regression

    lr.theta = np.array([[250], [-220], [5000]])
    lr.run("Force theta")  # Run the regression
