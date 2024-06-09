import csv
import os
import json
import logging
import pandas as pd  # Add this line if not already present

def save_classification_report_to_csv(report, filename, classifier_name, config):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.reset_index(inplace=True)
    df_report.rename(columns={'index': 'Class'}, inplace=True)
    df_report['Classifier'] = classifier_name
    for key, value in config.items():
        df_report[f"Config_{key}"] = value
    df_report.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

def initialize_csv_log(file_path):
    headers = [
        'IMG_WIDTH', 'IMG_HEIGHT', 'BATCH_SIZE', 'EPOCHS', 'SEED', 'CNN_ACCURACY', 'CNN_LOSS',  
        'KNN_ACCURACY', 'SVM_ACCURACY', 'KNN_LATENCY', 'SVM_LATENCY',  
        'RF_ACCURACY', 'RF_LATENCY', 'DT_ACCURACY', 'DT_LATENCY', 'LR_ACCURACY', 'LR_LATENCY',
        'DynaDetect_ACCURACY', 'DynaDetect_LATENCY', 'KNN_MSE', 'DynaDetect_MSE',
        'TOTAL_EXECUTION_TIME', 'TRAINING_TIME', 'EVALUATION_TIME', 'OVERALL_PERFORMANCE_SCORE',  
        'DATASET', 'EVALUATION', 'NUM_POISONED_IMAGES', 'TOTAL_IMAGES', 'POISON_PERCENT',
        'TRAIN_FEATURE_EXTRACTION_TIME', 'TRAIN_AVG_LATENCY_PER_IMAGE', 
        'TEST_FEATURE_EXTRACTION_TIME', 'TEST_AVG_LATENCY_PER_IMAGE','ATTACK_TYPE'
    ]

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def log_run_to_csv(file_path, config, history, accuracies, evaluation_times, training_time, evaluation_time,  
                   overall_performance_score, total_execution_time, num_poisoned_images, total_train_images, total_test_images, evaluation, attack_type, mse_values):
    if hasattr(history, 'history'):
        history_data = history.history
    else:
        history_data = history
    
    run_data = [
        config['IMG_WIDTH'], config['IMG_HEIGHT'], config['BATCH_SIZE'], config['EPOCHS'], config['SEED'],
        history_data['accuracy'][-1], history_data['loss'][-1], 
        accuracies['KNN'], accuracies.get('SVM', 0), 
        evaluation_times.get('KNN', 0), evaluation_times.get('SVM', 0), 
        accuracies.get('Random Forest', 0), evaluation_times.get('Random Forest', 0), 
        accuracies.get('Decision Tree', 0), evaluation_times.get('Decision Tree', 0), 
        accuracies.get('Logistic Regression', 0), evaluation_times.get('Logistic Regression', 0),
        accuracies.get('DynaDetect', 0), evaluation_times.get('DynaDetect', 0), 
        mse_values.get('KNN', 0), mse_values.get('DynaDetect', 0),
        total_execution_time, training_time, evaluation_time, 
        overall_performance_score, config['DATASET'], evaluation, num_poisoned_images,
        total_train_images + total_test_images,  
        (num_poisoned_images / (total_train_images + total_test_images)) * 100 if total_train_images + total_test_images > 0 else 0, 
        evaluation_times.get('train_feature_extraction_time', 0), evaluation_times.get('train_avg_latency_per_image', 0),
        evaluation_times.get('test_feature_extraction_time', 0), evaluation_times.get('test_avg_latency_per_image', 0),
        attack_type
    ]

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(run_data)
