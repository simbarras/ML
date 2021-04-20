# Import
import numpy as np  # Matrix
from sklearn.datasets import make_regression  # Random dataset
import matplotlib.pyplot as plt  # Graphic


# Linear regression module
class Lr:
    # Hyper parameters
    learning_rate = 0.01
    n_iteration = 100
    n_samples = 100

    # Configuration
    n_param_more = 0
    use_normal_vector = False
    verbose = False

    # dataset
    x = None
    y = None
    n_features = 1

    def run(self, name=""):  # return the coef and the final theta
        self.y = self.y.reshape(self.y.shape[0], 1)

        X = self.make_X(self.n_features + self.n_param_more)  # X -> arguments input [[a, 1],[b, 1]...]

        if not self.use_normal_vector:

            theta = np.random.randn(X.shape[1], 1)  # Matrix with the arguments of the function

            # Learn
            theta_final, cost_history = self.gradient_descent(X, self.y, theta, self.learning_rate, self.n_iteration)
        else:
            theta_final = self.normal_vector(self.X, self.y)

        predictions = self.model(X, theta_final)  # Make predictions
        coef = self.coef_determination(self.y, predictions)  # Compute the coef / 1

        if self.verbose:
            print("\nStats: Linear regression ({})".format(name))  # Show Stats
            print("Use normal vector: " + str(self.use_normal_vector))
            print("Samples: " + str(self.n_samples))
            print("Features: " + str(self.n_features))
            print("More parameters: " + str(self.n_param_more))
            if not self.use_normal_vector:
                print("Iteration: " + str(self.n_iteration))
                print("Learning rate: " + str(self.learning_rate))
            print("y: " + str(self.y.shape))
            print("x: " + str(self.x.shape))
            print("Theta: " + str(theta_final.shape))
            print("Theta final: " + str(theta_final))

            # Show coef
            print("Coef: {0:9.3f}/1 ({0})".format(coef))

            # Show graphs
            for i in range(0, self.n_features):
                plt.scatter(self.x[:, i], self.y)
                plt.scatter(self.x[:, i], predictions, c='r')
                plt.show()

            #  Show cost evolution
            if not self.use_normal_vector:
                plt.plot(range(self.n_iteration), cost_history)
                plt.show()

        return theta_final, coef

    """=================================================================================================================
    ¦¦¦ Machine Learning FUNCTION                                                                                    ¦¦¦
    ================================================================================================================="""

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

    def normal_vector(self, x, y):
        return np.linalg.inv(x.T.dot(x)).dot((x.T.dot(y)))

    def coef_determination(self, y, pred):
        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v

    """=================================================================================================================
    ¦¦¦ CONSTRUCTION FUNCTION                                                                                        ¦¦¦
    ================================================================================================================="""

    def make_X(self, n_param):
        if n_param <= 2:
            return np.hstack((self.x, np.ones((self.x.shape[0], 1))))
        else:
            return np.hstack((self.x ** (n_param - 1), self.make_X(n_param - 1)))

    def random_dataset(self, noise):
        self.x, self.y = make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                         noise=noise)  # x -> inputs  y -> outputs
        self.y = self.y.reshape(self.y.shape[0], 1)
        return self.x, self.y
