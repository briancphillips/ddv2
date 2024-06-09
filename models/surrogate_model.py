import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class SurrogateModel:
    def __init__(self, input_shape, num_classes, model_path='surrogate_model_gtsrb.h5'):
        self.model_path = model_path
        self.model = self._build_model(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        model = Sequential([
            Dense(64, input_shape=(input_shape,), activation='relu'),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0009), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    def save_model(self):
        self.model.save(self.model_path)
        print(f"Surrogate model saved to {self.model_path}")

    def load_model(self):
        self.model = load_model(self.model_path)
        print(f"Surrogate model loaded from {self.model_path}")

    def model_exists(self):
        return os.path.exists(self.model_path)

    @staticmethod
    def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, iters=10):
        perturbed_X = tf.convert_to_tensor(X, dtype=tf.float32)
        original_X = perturbed_X 
        y = tf.convert_to_tensor(y, dtype=tf.int64) 

        for _ in range(iters):
            with tf.GradientTape() as tape:
                tape.watch(perturbed_X)
                predictions = model(perturbed_X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions, from_logits=True)
            gradients = tape.gradient(loss, perturbed_X)
            perturbation = alpha * tf.sign(gradients)
            perturbed_X += perturbation
            perturbed_X = tf.clip_by_value(original_X + tf.clip_by_value(perturbed_X - original_X, -epsilon, epsilon), 0, 1)

        num_poisoned_images = len(X)
        return perturbed_X.numpy(), num_poisoned_images

    @staticmethod
    def label_flipping_attack(y, num_classes, num_to_flip=100, source_class=None, target_class=None):
        y_flipped = y.copy()

        if source_class is not None and target_class is not None:
            if source_class != target_class:
                source_indices = np.where(y == source_class)[0]
                num_to_flip = min(num_to_flip, len(source_indices))
                flip_indices = np.random.choice(source_indices, num_to_flip, replace=False)
                y_flipped[flip_indices] = target_class
            else:
                print("source_class and target_class cannot be the same. No flipping performed.")
        elif target_class is not None:
            num_to_flip = min(num_to_flip, len(y))
            flip_indices = np.random.choice(len(y), num_to_flip, replace=False)
            y_flipped[flip_indices] = target_class
        else:
            num_to_flip = min(num_to_flip, len(y))
            flip_indices = np.random.choice(len(y), num_to_flip, replace=False)
            for idx in flip_indices:
                current_label = y_flipped[idx]
                new_label = np.random.randint(0, num_classes)
                while new_label == current_label:
                    new_label = np.random.randint(0, num_classes)
                y_flipped[idx] = new_label

        return y_flipped

    @staticmethod
    def gradient_ascent_attack(model, X, y, epsilon=0.1, alpha=0.01, iters=10):
        perturbed_X = tf.convert_to_tensor(X, dtype=tf.float32)
        original_X = perturbed_X 
        y = tf.convert_to_tensor(y, dtype=tf.int64)

        for _ in range(iters):
            with tf.GradientTape() as tape:
                tape.watch(perturbed_X)
                predictions = model(perturbed_X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions, from_logits=True)
            gradients = tape.gradient(loss, perturbed_X)
            perturbed_X += alpha * gradients
            perturbed_X = tf.clip_by_value(original_X + tf.clip_by_value(perturbed_X - original_X, -epsilon, epsilon), 0, 1)

        num_poisoned_images = len(X)
        return perturbed_X.numpy(), num_poisoned_images
