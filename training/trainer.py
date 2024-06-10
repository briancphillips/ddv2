import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.dynadetect import DynaDetect  # Add this line

class ModelTrainer:
    def __init__(self, model, train_generator, validation_generator, config, steps_per_epoch_train, steps_per_epoch_val):
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.config = config
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_val = steps_per_epoch_val
        self.training_time = 0

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        print(f"Steps per epoch (train): {self.steps_per_epoch_train}")
        print(f"Steps per epoch (validation): {self.steps_per_epoch_val}")

        # Ensure the dataset repeats
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self.train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, self.config['IMG_WIDTH'], self.config['IMG_HEIGHT'], 3], [None, 43])
        ).repeat()

        val_dataset = tf.data.Dataset.from_generator(
            lambda: self.validation_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, self.config['IMG_WIDTH'], self.config['IMG_HEIGHT'], 3], [None, 43])
        ).repeat()
        
        training_start_time = time.time()
        history = self.model.fit(
            train_dataset,
            steps_per_epoch=self.steps_per_epoch_train,
            epochs=self.config['EPOCHS'],
            validation_data=val_dataset,
            validation_steps=self.steps_per_epoch_val,
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        return history

    def train_dynadetect(self, X_train, y_train):
        print(f"Training DynaDetect with config: {self.config}")
        dynadetect = DynaDetect(self.config)
        dynadetect.fit(X_train, y_train)
        return dynadetect
