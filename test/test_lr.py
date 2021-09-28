from lr import Lr as regressor

def test_random():
    lr = regressor()

    lr.learning_rate = 0.0001
    lr.n_iteration = 1000
    lr.n_samples = 5
    lr.n_param_more = 2
    lr.verbose = True

    lr.n_features = 1

    
    lr.random_dataset(10)

    lr.run("Grad")  # Run the regression

    lr.run("Force theta")  # Run the regression
