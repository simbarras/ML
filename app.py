from ML.lr.lr import Lr
from ML.lr.lr_multiple import LrMultiple
from ML.lr.lr_simple import LrSimple

if __name__ == '__main__':
    # lr = LrSimple()
    #lr = LrMultiple()
    lr = Lr()

    lr.learning_rate = 0.01
    lr.n_iteration = 100
    lr.n_samples = 100
    lr.n_features = 1
    lr.n_param_more = 0
    lr.verbose = True

    lr.randomDataset()
    lr.run()  # Run the regression
