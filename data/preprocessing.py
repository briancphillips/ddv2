import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def _is_valid_directory(self, d, data_dir):
        return os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.') and d != '.ipynb_checkpoints'

    def _clean_directory(self, data_dir):
        for root, dirs, files in os.walk(data_dir):
            for dir_name in dirs:
                if dir_name.startswith('.') or dir_name == '.ipynb_checkpoints':
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path, ignore_errors=True)

    def load_and_preprocess_data(self, train_data_dir, validation_data_dir):
        # Clean up directories
        self._clean_directory(train_data_dir)
        self._clean_directory(validation_data_dir)

        train_classes = [d for d in os.listdir(train_data_dir) if self._is_valid_directory(d, train_data_dir)]
        val_classes = [d for d in os.listdir(validation_data_dir) if self._is_valid_directory(d, validation_data_dir)]

        print(f"Training classes: {train_classes}")
        print(f"Validation classes: {val_classes}")

        num_train_classes = len(train_classes)
        num_val_classes = len(val_classes)

        print(f"Number of classes in training data: {num_train_classes}")
        print(f"Number of classes in validation data: {num_val_classes}")

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.12,
            height_shift_range=0.12,
            shear_range=10,
            zoom_range=0.1,
            brightness_range=[0.9, 1.1],
            channel_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.config['IMG_WIDTH'], self.config['IMG_HEIGHT']),
            batch_size=self.config['BATCH_SIZE'],
            class_mode='categorical',
            seed=self.config['SEED'],
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.config['IMG_WIDTH'], self.config['IMG_HEIGHT']),
            batch_size=self.config['BATCH_SIZE'],
            class_mode='categorical',
            seed=self.config['SEED'],
            shuffle=True
        )

        print(f"Class indices (train): {train_generator.class_indices}")
        print(f"Class indices (validation): {validation_generator.class_indices}")

        steps_per_epoch_train = train_generator.samples // self.config['BATCH_SIZE']
        steps_per_epoch_val = validation_generator.samples // self.config['BATCH_SIZE']
        print(f"Steps per epoch (train): {steps_per_epoch_train}")
        print(f"Steps per epoch (validation): {steps_per_epoch_val}")

        return train_generator, validation_generator, steps_per_epoch_train, steps_per_epoch_val
