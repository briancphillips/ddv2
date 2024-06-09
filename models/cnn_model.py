import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class CNNModel:
    def __init__(self, config):
        self.config = config
        self.model_path = 'cnn_model_gtsrb.h5'
        self.history_path = 'cnn_model_gtsrb_history.json'  # Consistent filename for history

    def build_model(self):
        if os.path.exists(self.model_path):
            print("Loading existing CNN model...")
            model = load_model(self.model_path)
        else:
            print("Building new CNN model...")
            tf.random.set_seed(self.config['SEED'])
            tf.config.experimental.enable_op_determinism()
            model = Sequential()

            weight_decay = 0.0001

            model.add(InputLayer(input_shape=(self.config['IMG_WIDTH'], self.config['IMG_HEIGHT'], 3)))
            model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.2))
            
            model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.3))
            
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.4))
            
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.5))
            
            model.add(Flatten())
            model.add(Dense(43, activation='softmax'))

            model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def save_model(self, model):
        model.save(self.model_path)
        print("CNN model saved.")

    def save_history(self, history):
        history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        with open(self.history_path, 'w') as f:
            json.dump(history_dict, f)
        print("Training history saved.")
