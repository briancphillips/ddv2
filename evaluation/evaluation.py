import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from models.surrogate_model import SurrogateModel
from training.trainer import ModelTrainer
from utils.helpers import save_classification_report_to_csv

class ClassifierEvaluation:
    def __init__(self, evaluation, config):
        self.evaluation = evaluation
        self.config = config

    def evaluate_classifiers(self, X_train, y_train, X_test, y_test, evaluation_times, surrogate_model=None, attack_type=None, num_classes=None, source_class=None, target_class=None):
        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            # "SVM": SVC(),
            #"Random Forest": RandomForestClassifier(),
            # "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='saga', C=1.0, warm_start=True, tol=1e-4)
        }

        accuracies = {}
        classification_reports = {}
        mse_values = {}

        for name, clf in classifiers.items():
            print(f'Evaluating with {name}')
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            end_time = time.time()
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[name] = accuracy
            evaluation_times[name] = end_time - start_time
            mse = mean_squared_error(y_test, y_pred)
            mse_values[name] = mse

            report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
            classification_reports[name] = report

        num_poisoned_images = 0
        dynadetect_predictions = np.array([])  # Initialize as empty array

        if self.evaluation == "Poisoned" and surrogate_model is not None:
            if attack_type == "pgd":
                X_test_scaled, num_poisoned_images = SurrogateModel.pgd_attack(surrogate_model, X_test, y_test)
            elif attack_type == "label_flip":
                y_test_poisoned = SurrogateModel.label_flipping_attack(y_test, num_classes, num_to_flip=10000, source_class=source_class, target_class=target_class)
                X_test_scaled = X_test
                num_poisoned_images = np.sum(y_test_poisoned != y_test)
            else:
                X_test_scaled = X_test

            print("Evaluating DynaDetect...")
            steps_per_epoch_train = self.config['steps_per_epoch_train']
            steps_per_epoch_val = self.config['steps_per_epoch_val']
            dynadetect = ModelTrainer(None, None, None, self.config, steps_per_epoch_train, steps_per_epoch_val).train_dynadetect(X_train, y_train)
            dynadetect_predictions, dynadetect_mse_values = dynadetect.predict(X_test)
            dynadetect_accuracy = accuracy_score(y_test, dynadetect_predictions)
            accuracies["DynaDetect"] = dynadetect_accuracy
            mse_values["DynaDetect"] = dynadetect_mse_values

            # Debug statements to check invalid predictions
            print("Before replacement:")
            print("Unique labels in y_test:", np.unique(y_test))
            print("Unique labels in dynadetect_predictions:", np.unique(dynadetect_predictions))

            # Replace or remove invalid predictions (-1)
            invalid_mask = dynadetect_predictions == -1
            if np.any(invalid_mask):
                most_frequent_class = np.bincount(dynadetect_predictions[dynadetect_predictions != -1]).argmax()
                dynadetect_predictions[invalid_mask] = most_frequent_class

            # Debug statements to check if replacements were successful
            print("After replacement:")
            print("Unique labels in dynadetect_predictions:", np.unique(dynadetect_predictions))

            # Add DynaDetect to classification reports
            report = classification_report(y_test, dynadetect_predictions, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
            classification_reports["DynaDetect"] = report

        print(f"Accuracies: {accuracies}")
        print(f"Evaluation times: {evaluation_times}")
        print(f"Number of poisoned images: {num_poisoned_images}")
        print(f"Classification reports: {classification_reports}")
        print(f"MSE values: {mse_values}")

        return accuracies, evaluation_times, num_poisoned_images, classification_reports, mse_values
