from ML.lr.lr_multiple import LrMultiple
from ML.lr.lr_simple import LrSimple

if __name__ == '__main__':
    # lr = LrSimple()
    lr = LrMultiple()

    lr.run()  # Run the regression
