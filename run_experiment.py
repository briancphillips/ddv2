import datetime
from pipeline.pipeline import Pipeline
from utils.helpers import initialize_csv_log


base_config = {
    'IMG_WIDTH': 30,
    'IMG_HEIGHT': 30,
    'BATCH_SIZE': 128,
    'EPOCHS': 2,
    'SEED': 42,
    'DATASET': 'GTSRB',
    'train_data_dir': './datasets/gtsrb/train',
    'validation_data_dir': './datasets/gtsrb/val',
    'k': 5,
    'num_iterations': 10,
    'learning_rate': 0.001,
    'regularization': 0.01,
    'batch_size': 128,
    'num_models': 5,
    'mse_threshold': 0.1,
}

configurations = [
    # PGD attack with poisoned evaluation
    {**base_config, 'attack_type': 'pgd', 'evaluation': 'Poisoned', 'BATCH_SIZE': 16, 'source_class': 3, 'target_class': 7},
    # PGD attack with clean evaluation
    {**base_config, 'attack_type': 'pgd', 'evaluation': 'Clean', 'BATCH_SIZE': 32, 'source_class': 3, 'target_class': 7},
    # # Label flipping attack with poisoned evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Poisoned', 'BATCH_SIZE': 64, 'source_class': 2, 'target_class': 8},
    # # Label flipping attack with clean evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Clean', 'BATCH_SIZE': 64, 'source_class': 2, 'target_class': 8},
    # # Gradient ascent attack with poisoned evaluation
    {**base_config, 'attack_type': 'gradient_ascent', 'evaluation': 'Poisoned', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': None},
    # # Gradient ascent attack with clean evaluation
    {**base_config, 'attack_type': 'gradient_ascent', 'evaluation': 'Clean', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': None},
    
    # Random to target attack with poisoned evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Poisoned', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': 9},
    # Random to target attack with clean evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Clean', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': 9},
    # Random to random attack with poisoned evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Poisoned', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': None},
    # Random to random attack with clean evaluation
    {**base_config, 'attack_type': 'label_flip', 'evaluation': 'Clean', 'BATCH_SIZE': 32, 'source_class': None, 'target_class': None},
]


def automate_pipeline_execution(pipeline, configurations):
    results = []
    for config in configurations:
        pipeline.update_config(config)
        result = pipeline.run_pipeline(evaluation=config['evaluation'], attack_type=config['attack_type'])
        results.append(result)
    return results

pipeline = Pipeline('config/config.yaml')
initialize_csv_log('experiment_logs.csv')
start_time = datetime.datetime.now()
results = automate_pipeline_execution(pipeline, configurations)
end_time = datetime.datetime.now()

elapsed_time = end_time - start_time
minutes = elapsed_time.seconds // 60
seconds = elapsed_time.seconds % 60
print(f"Elapsed Time: {minutes} minutes and {seconds} seconds")
