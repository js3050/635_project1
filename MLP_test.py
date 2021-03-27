import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

def load_tf_model(path):
    return tf.keras.models.load_model(path)
if __name__ == "__main__":
    model_path = "mlp"
    testing_data = pd.read_csv("dataset/test.csv")

    model = load_tf_model(model_path)
    predictions = model.predict(testing_data)



