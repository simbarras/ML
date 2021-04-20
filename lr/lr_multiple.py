# Import
import numpy as np  # Matrix
from sklearn.datasets import make_regression  # Random dataset
import matplotlib.pyplot as plt  # Graphic


# Linear regression multiple
# https://github.com/MachineLearnia/Regression-lineaire-numpy/blob/master/R%C3%A9gression%20Lin%C3%A9aire%20Multiple.ipynb
class LrMultiple:
    # Hyper parameter
    learning_rate = 0.1
    n_iteration = 100

    # Configuration
    n_features = 1
    n_param = 2

    def run(self):
        x, y = make_regression(n_samples=100, n_features=self.n_features, noise=10)  # x -> inputs  y -> outputs
        y = y.reshape(y.shape[0], 1)

        X = np.hstack((x, np.ones(x.shape)))  # X -> arguments input [[a, 1],[b, 1]...]
        for i in range(2, self.n_param):
            X = np.hstack((X * x, np.ones(x.shape)))
            # print(X)

        theta = np.random.randn(X.shape[1], 1)  # Matrix with the arguments of the function

        # Learn
        theta_final, cost_history = self.gradient_descent(X, y, theta, self.learning_rate, self.n_iteration)

        predictions = self.model(X, theta_final)

        print("Stats: Linear regression multiple")  # Show Stats
        print("Iteration: " + str(self.n_iteration))
        print("Learning rate: " + str(self.learning_rate))
        print("Features: " + str(self.n_features))
        print("Parameters: " + str(self.n_param))
        print("x: " + str(x.shape))
        print("y: " + str(y.shape))
        print("X: " + str(X.shape))
        print("Theta: " + str(theta.shape))
        print("Theta final: " + str(theta_final))

        # Show graph
        plt.scatter(x, y)
        plt.plot(x, predictions, c='r')
        plt.show()

        #  Show cost evolution
        plt.plot(range(self.n_iteration), cost_history)
        plt.show()

        # Show coef /1
        coef = self.coef_determination(y, predictions)
        print("\nCoef: {0:9.3f}/1 ({0})".format(coef))

    def model(self, X, theta):
        return X.dot(theta)

    def cost(self, X, y, theta):
        m = len(y)
        return 1 / (2 * m) * np.sum((self.model(X, theta) - y) ** 2)

    def grad(self, X, y, theta):
        m = len(y)
        return 1 / m * X.T.dot(self.model(X, theta) - y)

    def gradient_descent(self, X, y, theta, learning_rate, n_iteration):
        cost_history = np.zeros(n_iteration)
        for i in range(0, n_iteration):
            theta = theta - learning_rate * self.grad(X, y, theta)
            cost_history[i] = self.cost(X, y, theta)
        return theta, cost_history

    def coef_determination(self, y, pred):
        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
