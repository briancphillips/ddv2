import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from models.surrogate_model import SurrogateModel
from training.trainer import ModelTrainer
from utils.helpers import save_classification_report_to_csv
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import joblib
from tensorflow.keras.models import load_model
from keras.models import load_model


# Define TensorFlow Decision Forests model wrappers
class TensorFlowSVM:
    def __init__(self, input_dim, num_classes, learning_rate=0.01, lambda_param=0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(self.num_classes, kernel_regularizer=tf.keras.regularizers.l2(self.lambda_param))
        ])
        return model
    
    def fit(self, X_train, y_train, epochs=10):
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                           loss='hinge',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

class TensorFlowLogisticRegression:
    def __init__(self, input_dim, num_classes, learning_rate=0.01):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=0)

    def predict(self, X_test):
        return tf.argmax(self.model.predict(X_test), axis=1).numpy()

class TensorFlowDecisionTree:
    def __init__(self, batch_size):
        self.model = tfdf.keras.CartModel(check_dataset=False)
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        self.model.fit(dataset)

    def predict(self, X_test):
        dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(self.batch_size)
        return self.model.predict(dataset).argmax(axis=1)

class TensorFlowRandomForest:
    def __init__(self, batch_size):
        self.model = tfdf.keras.RandomForestModel(check_dataset=False)
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        self.model.fit(dataset)

    def predict(self, X_test):
        dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(self.batch_size)
        return self.model.predict(dataset).argmax(axis=1)
        
class ClassifierEvaluation:
    def __init__(self, evaluation, config):
        self.evaluation = evaluation
        self.config = config

    def evaluate_classifiers(self, X_train, y_train, X_test, y_test, evaluation_times, surrogate_model=None, attack_type=None, num_classes=None, source_class=None, target_class=None):
        input_dim = X_train.shape[1]
        batch_size = self.config.get('BATCH_SIZE', 32)  # Default to 32 if not specified
    
        classifiers = {
            "KNN": ('knn', KNeighborsClassifier(n_neighbors=5)),
            "SVM": ('tf_svm', TensorFlowSVM(input_dim=input_dim, num_classes=num_classes)),
            "Random Forest": ('tf_random_forest', TensorFlowRandomForest(batch_size=batch_size)),
            "Decision Tree": ('tf_decision_tree', TensorFlowDecisionTree(batch_size=batch_size)),
            "Logistic Regression": ('tf_logistic_regression', TensorFlowLogisticRegression(input_dim=input_dim, num_classes=num_classes))
        }
    
        accuracies = {}
        classification_reports = {}
        mse_values = {}
    
        # Evaluate classifiers based on the evaluation type
        X_test_evaluation = X_test
        y_test_evaluation = y_test
    
        if self.evaluation == "Poisoned" and surrogate_model is not None:
            if attack_type == "pgd":
                X_test_evaluation, num_poisoned_images = SurrogateModel.pgd_attack(surrogate_model, X_test, y_test)
            elif attack_type == "label_flip":
                y_test_evaluation = SurrogateModel.label_flipping_attack(y_test, num_classes, num_to_flip=10000, source_class=source_class, target_class=target_class)
                X_test_evaluation = X_test
                num_poisoned_images = np.sum(y_test_evaluation != y_test)
            else:
                X_test_evaluation = X_test
    
        for name, (file_prefix, clf) in classifiers.items():
            print(f'Evaluating with {name}')
            model_path = f'{file_prefix}_model'
    
            if os.path.exists(model_path):
                if isinstance(clf, (TensorFlowRandomForest, TensorFlowDecisionTree)):
                    clf.model.load_weights(model_path)
                elif isinstance(clf, (TensorFlowSVM, TensorFlowLogisticRegression)):
                    clf.model = tf.keras.models.load_model(f"{model_path}.keras")
                else:
                    clf = joblib.load(f"{model_path}.joblib")
            else:
                start_time = time.time()
                clf.fit(X_train, y_train)
                end_time = time.time()
                if isinstance(clf, (TensorFlowRandomForest, TensorFlowDecisionTree)):
                    clf.model.save_weights(model_path)
                elif isinstance(clf, (TensorFlowSVM, TensorFlowLogisticRegression)):
                    clf.model.save(f"{model_path}.keras")
                else:
                    joblib.dump(clf, f"{model_path}.joblib")
    
                evaluation_times[name] = end_time - start_time
    
            y_pred = clf.predict(X_test_evaluation)
            accuracy = accuracy_score(y_test_evaluation, y_pred)
            accuracies[name] = accuracy
            mse = mean_squared_error(y_test_evaluation, y_pred)
            mse_values[name] = mse
    
            report = classification_report(y_test_evaluation, y_pred, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
            classification_reports[name] = report
    
        # Evaluate DynaDetect on clean data
        print("Evaluating DynaDetect on clean data...")
        steps_per_epoch_train = self.config['steps_per_epoch_train']
        steps_per_epoch_val = self.config['steps_per_epoch_val']
        dynadetect = ModelTrainer(None, None, None, self.config, steps_per_epoch_train, steps_per_epoch_val).train_dynadetect(X_train, y_train)
    
        # Evaluate DynaDetect on the test data (clean or poisoned)
        dynadetect_predictions, dynadetect_mse_values = dynadetect.predict(X_test_evaluation)
        dynadetect_accuracy = accuracy_score(y_test_evaluation, dynadetect_predictions)
        accuracies["DynaDetect"] = dynadetect_accuracy
        mse_values["DynaDetect"] = dynadetect_mse_values

    
        # Debug statements to check invalid predictions
        print("Before replacement:")
        print("Unique labels in y_test_evaluation:", np.unique(y_test_evaluation))
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
        report = classification_report(y_test_evaluation, dynadetect_predictions, target_names=[f'Class {i}' for i in range(num_classes)], output_dict=True)
        classification_reports["DynaDetect"] = report
    
        print(f"Accuracies: {accuracies}")
        print(f"Evaluation times: {evaluation_times}")
        print(f"Number of poisoned images: {num_poisoned_images if self.evaluation == 'Poisoned' else 0}")
        print(f"Classification reports: {classification_reports}")
        print(f"MSE values: {mse_values}")
    
        return accuracies, evaluation_times, num_poisoned_images if self.evaluation == 'Poisoned' else 0, classification_reports, mse_values