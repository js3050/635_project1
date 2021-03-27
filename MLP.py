import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


class MLP:
    model = None
    dataset_path = None
    dataset = None
    X, Y = None, None
    X_train, X_test, y_train, y_test = None, None, None, None
    epochs = 10

    def __init__(self, dataset_path, epochs=10):
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.dataset = pd.read_csv(self.dataset_path)
        self.process_data()
        self.initialize_model()
        self.train(self.epochs)
        loss, accuracy = self.evaluate()
        print(f"Loss : {loss} Accuracy : {accuracy}")

    def process_data(self):
        self.Y = self.dataset['label']
        self.X = self.dataset.drop('label', axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y)

        # normalize data
        self.X_train = tf.keras.utils.normalize(self.X_train)
        self.X_test = tf.keras.utils.normalize(self.X_test)

    def initialize_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epochs):
        self.model.fit(self.X_train, self.y_train, epochs=epochs)


    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss,accuracy

    def save_model(self, model_name="mlp"):
        self.model.save(model_name)

if __name__ == "__main__":
    training_path = "dataset/train.csv"
    mlp_model = MLP(training_path)
    mlp_model.save_model()


