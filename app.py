from ML.lr.lr import Lr
from ML.lr.lr_multiple import LrMultiple
from ML.lr.lr_simple import LrSimple

if __name__ == '__main__':
    # lr = LrSimple()
    #lr = LrMultiple()
    lr = Lr()

    lr.verbose = True

    lr.n_features = 2
    lr.randomDataset()
    #lr.run()  # Run the regression
    lr.n_param = 3
    lr.run()
