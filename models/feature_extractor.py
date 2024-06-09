import tensorflow as tf
from tensorflow.keras.layers import Layer

class FeatureExtractor(Layer):
    def __init__(self, model):  
        super(FeatureExtractor, self).__init__()
        self.feature_extraction_layer = model.get_layer(index=-2) 

    def call(self, inputs):
        return self.feature_extraction_layer(inputs)  
        
    def extract_features_and_measure_latency(self, generator, total_samples, subset_size, evaluation_times, phase):
        start_time = time.time()
        features, labels = []

        if subset_size is not None:
            steps_per_epoch = min(subset_size // generator.batch_size, total_samples // generator.batch_size)
        else:
            steps_per_epoch = total_samples // generator.batch_size

        for i in range(steps_per_epoch):
            x_batch, y_batch = next(generator)
            features_batch = self.feature_extractor.predict(x_batch, verbose=0)
            features.append(features_batch)
            labels.append(y_batch)

        total_time = time.time() - start_time
        avg_latency_per_image = total_time / (steps_per_epoch * generator.batch_size)
        evaluation_times[f'{phase}_feature_extraction_time'] = total_time
        evaluation_times[f'{phase}_avg_latency_per_image'] = avg_latency_per_image

        labels = np.concatenate(labels)
        labels = np.argmax(labels, axis=1)

        return np.concatenate(features), labels
