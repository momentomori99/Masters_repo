import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

class Preprocessing:
    def __init__(self):
        self.scaler = MinMaxScaler() 
        self.r_min = 1
        self.r_max = 50
        self.rate_scale = 5
        self.noise = 0.1

    def import_iris_dataset(self):
        """
        Import the iris dataset and preprocess it.

        inputs:
            r_min: minimum rate in Hz (int)
            r_max: maximum rate in Hz (int)

        outputs:
            r: normalized rates (numpy array)
            y: labels (numpy array)
        """
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        X_normalized = self.scaler.fit_transform(X)
        r = self.r_min + X_normalized * (self.r_max - self.r_min)

        summary = ""
        summary += f"Dataset: iris\n"
        summary += f"Number of samples: {len(X)}\n"
        summary += f"Number of features: {len(X[0])}\n"
        summary += f"Noise: 0\n"
        summary += f"r_min: {self.r_min}\n"
        summary += f"r_max: {self.r_max}\n"
        summary += f"rate_scale: {self.rate_scale}\n"
        print(summary)
        return summary, r*self.rate_scale, y
    
    def import_moon_dataset(self, plot = True):
        moons = make_moons(n_samples=150, noise=self.noise)
        X = np.array(moons[0])
        y = np.array(moons[1])
        # Order X and y by class: all class 0s first, then class 1s
        idx_class0 = np.where(y == 0)[0]
        idx_class1 = np.where(y == 1)[0]
        idx_sorted = np.concatenate([idx_class0, idx_class1])
        X = X[idx_sorted]
        y = y[idx_sorted]

        if plot:
            for i in range(len(y)):
                if y[i] == 0:
                    plt.plot(X[i][0], X[i][1], "o", color="r")
                else:
                    plt.plot(X[i][0], X[i][1], "o", color="b")
            plt.show()
        
        X_normalized = self.scaler.fit_transform(X)
        r = self.r_min + X_normalized * (self.r_max - self.r_min)

        summary = ""
        summary += f"Dataset: moons\n"
        summary += f"Number of samples: {len(X)}\n"
        summary += f"Number of features: {len(X[0])}\n"
        summary += f"Noise: {self.noise}\n"
        summary += f"r_min: {self.r_min}\n"
        summary += f"r_max: {self.r_max}\n"
        summary += f"rate_scale: {self.rate_scale}\n"
        print(summary)
        return summary, r*self.rate_scale, y    
    
    def import_circles_dataset(self, plot = False):
        circles = make_circles(n_samples=150, noise=self.noise)
        X = np.array(circles[0])
        y = np.array(circles[1])

        idx_class0 = np.where(y == 0)[0]
        idx_class1 = np.where(y == 1)[0]
        idx_sorted = np.concatenate([idx_class0, idx_class1])
        X = X[idx_sorted]
        y = y[idx_sorted]

        if plot:
            for i in range(len(y)):
                if y[i] == 0:
                    plt.plot(X[i][0], X[i][1], "o", color="r")
                else:
                    plt.plot(X[i][0], X[i][1], "o", color="b")
            plt.show()

        X_normalized = self.scaler.fit_transform(X)
        r = self.r_min + X_normalized * (self.r_max - self.r_min)
        summary = ""
        summary += f"Dataset: circles\n"
        summary += f"Number of samples: {len(X)}\n"
        summary += f"Number of features: {len(X[0])}\n"
        summary += f"Noise: {self.noise}\n"
        summary += f"r_min: {self.r_min}\n"
        summary += f"r_max: {self.r_max}\n"
        summary += f"rate_scale: {self.rate_scale}\n"
        return summary, r*self.rate_scale, y


if __name__ == "__main__":
    preprocessing = Preprocessing()
    # X, y = preprocessing.import_iris_dataset()
    # print(type(X))
    # print(type(y))
    preprocessing.import_moon_dataset()
    preprocessing.import_circles_dataset()
