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
    name = ""

    # dataset
    x = None
    y = None
    X = None
    n_features = 1
    theta = None

    # result
    theta_final = None
    coef = 0

    def run(self, name=""):  # return the coef and the final theta
        if name == "":
            name = self.name

        self.y = self.y.reshape(self.y.shape[0], 1)

        self.X = self.make_X(self.n_features + self.n_param_more)  # X -> arguments input [[a, 1],[b, 1]...]

        if not self.use_normal_vector:

            if self.theta is None:
                self.theta = np.random.randn(self.X.shape[1], 1)  # Matrix with the arguments of the function

            # Learn
            self.theta_final, cost_history = self.gradient_descent()
        else:
            self.theta = self.normal_vector()
            self.theta_final = self.theta

        self.coef = self.coef_determination()  # Compute the coef / 1

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
            print("X: " + str(self.X.shape))
            print("Theta: " + str(self.theta_final.shape))
            print("Theta final: " + str(self.theta_final))
            print("Coef: {0:9.3f}/1 ({0})".format(self.coef))

            # Show graphs
            for i in range(0, self.n_features):
                plt.scatter(self.x[:, i], self.y)
                plt.scatter(self.x[:, i], self.model(self.X), c='r')
                plt.show()

            #  Show cost evolution
            if not self.use_normal_vector:
                plt.plot(range(self.n_iteration), cost_history)
                plt.show()

        return self.theta_final, self.coef

    """=================================================================================================================
    ¦¦¦ Machine Learning FUNCTION                                                                                    ¦¦¦
    ================================================================================================================="""

    def model(self, x):
        return x.dot(self.theta)

    def cost(self):
        m = len(self.y)
        return 1 / (2 * m) * np.sum((self.model(self.X) - self.y) ** 2)

    def grad(self):
        m = len(self.y)
        return 1 / m * self.X.T.dot(self.model(self.X) - self.y)

    def gradient_descent(self):
        if self.verbose:
            cost_history = np.zeros(self.n_iteration)

        for i in range(0, self.n_iteration):
            self.theta = self.theta - self.learning_rate * self.grad()
            if self.verbose:
                cost_history[i] = self.cost()
        return self.theta, cost_history

    def normal_vector(self):
        return np.linalg.inv(self.X.T.dot(self.X)).dot((self.X.T.dot(self.y)))

    def coef_determination(self):
        pred = self.model(self.X)
        u = ((self.y - pred) ** 2).sum()
        v = ((self.y - self.y.mean()) ** 2).sum()
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
