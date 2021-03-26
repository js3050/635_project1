import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import learning_curve
from joblib import dump


class ClassicalModel:
    __slots__ = "dataset_folder", "training_data", "testing_data", "X_train", "X_test", "Y_train", "Y_test", "model"

    def __init__(self, dataset_folder, training_model=SVC()):
        """
        initialize dataset folder, parameters and read the dataset
        :param dataset_folder: path to dataset folder
        """
        self.training_data, self.testing_data, self.X_test, self.X_train, self.Y_test, self.Y_train = (None for _ in
                                                                                                       range(len(
                                                                                                           self.__slots__) - 2))
        self.model = None
        self.dataset_folder = dataset_folder
        self.read_data()
        self.generate_training_model(training_model)
        self.save_model()

    def read_data(self):
        """
        Read data from dataset folder and generate training and testing datasets
        :return:
        """
        self.training_data = pd.read_csv(self.dataset_folder + "/train.csv")
        Y = self.training_data['label']
        X = self.training_data.drop('label', axis=1)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y)

    def generate_training_model(self, model):
        self.model = model
        self._train()

    def _train(self):
        self.model.fit(self.X_train, self.Y_train)
        self.evaluate(self.X_test, self.Y_test)

    def evaluate(self, test_data, labels):
        X_test_predictions = self.model.predict(test_data)
        """
        Evaluate the model performance based on training split
        """
        precision, recall, fscore, support = precision_recall_fscore_support(X_test_predictions, labels)
        print(f"Precision {precision[0]} Recall {recall[0]} F Score {fscore[0]}")
        train_sizes, train_scores, valid_scores = learning_curve(self.model, self.X_train, self.Y_train)
        print(f"Train sizes : {train_sizes}, Train Scores {train_scores}, Valid Scores = {valid_scores}")

    def predict(self, input_data):
        return self.model.predict(input_data)

    def save_model(self, model_name="clf"):
        dump(self.model, model_name)


if __name__ == "__main__":
    data_model = ClassicalModel("dataset")
