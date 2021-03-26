import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

if __name__ == "__main__":
    test_data = pd.read_csv("dataset/test.csv")
    classifier = load("clf")
    pred = classifier.predict(test_data)
    for i in range(5) :
        plt.imshow(test_data.loc[i, :].to_numpy().reshape(28,28), label=pred[i])
        plt.show()
    pd.Series(pred).to_csv("svc_output.csv")