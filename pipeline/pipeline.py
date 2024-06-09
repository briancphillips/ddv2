import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from models.dynadetect import DynaDetect
from models.surrogate_model import SurrogateModel
from models.cnn_model import CNNModel
from models.feature_extractor import FeatureExtractor
from training.trainer import ModelTrainer
from evaluation.evaluation import ClassifierEvaluation
from utils.helpers import save_classification_report_to_csv, initialize_csv_log, log_run_to_csv
from data.preprocessing import DataPreprocessor
import yaml
import logging
import json
import pandas as pd

class Pipeline:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self._setup_logging()
        np.random.seed(self.config['SEED'])
        tf.random.set_seed(self.config['SEED'])

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def update_config(self, new_config):
        self.config.update(new_config)
        self.logger.info(f"Config updated to: {self.config}")

    def run_pipeline(self, evaluation='Clean', attack_type='pgd'):
        self.logger.info(f"Pipeline started with config: {self.config}")
        evaluation_times = {}

        data_preprocessor = DataPreprocessor(self.config)
        train_data_dir = self.config['train_data_dir']
        validation_data_dir = self.config['validation_data_dir']
        train_generator, validation_generator, steps_per_epoch_train, steps_per_epoch_val = data_preprocessor.load_and_preprocess_data(
            train_data_dir, validation_data_dir
        )

        self.config['steps_per_epoch_train'] = steps_per_epoch_train
        self.config['steps_per_epoch_val'] = steps_per_epoch_val

        cnn_model_builder = CNNModel(self.config)
        cnn_model = cnn_model_builder.build_model()
        model_trainer = ModelTrainer(cnn_model, train_generator, validation_generator, self.config, steps_per_epoch_train, steps_per_epoch_val)
        
        if not os.path.exists(cnn_model_builder.model_path):
            history = model_trainer.train_model()
            cnn_model_builder.save_model(cnn_model)
            cnn_model_builder.save_history(history)
        else:
            if os.path.exists(cnn_model_builder.history_path):
                history = self.load_history(cnn_model_builder.history_path)
            else:
                history = model_trainer.train_model()
                cnn_model_builder.save_history(history)

        feature_extractor = FeatureExtractor(cnn_model)  
        input_shape = cnn_model.input_shape[1:] 
        new_input = Input(shape=input_shape)  
        extracted_features = feature_extractor(new_input)
        feature_extractor_model = Model(inputs=new_input, outputs=extracted_features)

        total_train_samples = train_generator.samples
        total_val_samples = validation_generator.samples

        X_train, y_train = self.check_shapes(train_generator, feature_extractor_model, total_train_samples)
        X_test, y_test = self.check_shapes(validation_generator, feature_extractor_model, total_val_samples)

        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        num_classes = y_train.shape[1]

        surrogate_model_builder = SurrogateModel(X_train.shape[1], num_classes)
        surrogate_model = surrogate_model_builder.model

        if surrogate_model_builder.model_exists():
            surrogate_model_builder.load_model()
        else:
            surrogate_model_builder.fit(X_train, y_train_labels, epochs=self.config['EPOCHS'], batch_size=self.config['BATCH_SIZE'])
            surrogate_model_builder.save_model()

        classifier_evaluation = ClassifierEvaluation(evaluation, self.config)
        accuracies, evaluation_times, num_poisoned_images, classification_reports, mse_values = classifier_evaluation.evaluate_classifiers(
            X_train, y_train_labels, X_test, y_test_labels, evaluation_times, surrogate_model=surrogate_model, attack_type=attack_type, num_classes=num_classes
        )

        for clf_name, report in classification_reports.items():
            filename = f'./classification_report_{clf_name}_{attack_type}_{self.config["BATCH_SIZE"]}.csv'
            save_classification_report_to_csv(report, filename, clf_name, self.config)

        log_run_to_csv(
            'experiment_logs.csv',
            self.config,
            history,
            accuracies,
            evaluation_times,
            training_time=model_trainer.training_time,
            evaluation_time=0,
            overall_performance_score=0,
            total_execution_time=evaluation_times,
            num_poisoned_images=num_poisoned_images,
            total_train_images=total_train_samples,
            total_test_images=total_val_samples,
            evaluation=evaluation,
            attack_type=attack_type,
            mse_values=mse_values
        )

        self.logger.info("Pipeline completed")
        return {"config": self.config, "evaluation_times": evaluation_times}

    def save_classification_report_to_csv(self, report, filename, classifier_name, config):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.reset_index(inplace=True)
        df_report.rename(columns={'index': 'Class'}, inplace=True)
        df_report['Classifier'] = classifier_name
        for key, value in config.items():
            df_report[f"Config_{key}"] = value
        df_report.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    def load_history(self, filename):
        with open(filename, 'r') as f:
            history = json.load(f)
        return history

    def check_shapes(self, generator, model, total_samples):
        X, y = [], []
        steps_per_epoch = total_samples // generator.batch_size
        for i in range(steps_per_epoch):
            x_batch, y_batch = next(generator)
            features_batch = model.predict(x_batch, verbose=0)
            X.append(features_batch)
            y.append(y_batch)
        X = np.concatenate(X)
        y = np.concatenate(y)
        return X, y
