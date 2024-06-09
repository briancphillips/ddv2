import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

class DynaDetect:
    def __init__(self, config):
        self.k = config['k']
        self.num_iterations = config['num_iterations']
        self.learning_rate = config['learning_rate']
        self.regularization = config['regularization']
        self.batch_size = config['batch_size']
        self.num_models = config['num_models']
        self.mse_threshold = config.get('mse_threshold', float('inf'))
        self.models = []

    def build_model(self, input_shape, num_classes):
        input = Input(shape=input_shape)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(input)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(x)
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.regularization))(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        input_shape = self.X_train.shape[1:]
        num_classes = len(np.unique(y))
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        for _ in range(self.num_models):
            model = self.build_model(input_shape, num_classes)
            self.models.append(model)
        dataset = tf.data.Dataset.from_tensor_slices((self.X_train, y_one_hot))
        dataset = dataset.shuffle(buffer_size=len(self.X_train)).batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        for model in self.models:
            for _ in tqdm(range(self.num_iterations), desc="Training Iterations", leave=False):
                model.fit(dataset, epochs=1, steps_per_epoch=len(self.X_train) // self.batch_size, verbose=0)
                import gc
                gc.collect()

    def transform(self, X):
        transformed_features = [model.predict(X, batch_size=self.batch_size) for model in self.models]
        return np.concatenate(transformed_features, axis=1)

    def predict(self, X):
        X_transformed = self.transform(X)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(X_transformed.shape[0], X_transformed.shape[1]))
        X_transformed = pca.fit_transform(X_transformed)
        X_train_transformed = pca.transform(self.transform(self.X_train))
        predictions, mse_values = self._knn_predict(X_transformed, X_train_transformed)
        predictions[mse_values > self.mse_threshold] = -1  # Discard high MSE predictions
        return predictions, mse_values

    def _knn_predict(self, X_transformed, X_train_transformed):
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X_transformed, X_train_transformed)
        indices = np.argsort(distances, axis=1)[:, :self.k]
        nearest_labels = self.y_train[indices]
        batch_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_labels)
        batch_mse = np.mean(np.square(np.take_along_axis(distances, indices, axis=1)), axis=1)
        return batch_predictions, batch_mse
