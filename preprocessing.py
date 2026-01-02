from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

    def import_iris_dataset(self, r_min=7, r_max=50):
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
        
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)

        r_min = 7 
        r_max = 50 
        r = r_min + X_normalized * (r_max - r_min)

        return r, y
        



if __name__ == "__main__":
    preprocessing = Preprocessing()
    X, y = preprocessing.import_iris_dataset()
    print(type(X))
    print(type(y))