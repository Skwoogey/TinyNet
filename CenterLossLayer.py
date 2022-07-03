import tensorflow as tf
import tensorflow.keras.backend as K

class CenterLossLayer(tf.keras.layers.Layer):

    def __init__(self, class_count, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.class_count = class_count

    def build(self, input_shape):
        features_count = 1
        for i in range(1, len(input_shape[0])):
            features_count *= input_shape[0][i]

        self.centers = self.add_weight(name='centers',
                                       shape=(self.class_count, features_count),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nx2, x[1] is Nx1 label, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers))

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return tf.keras.backend.int_shape(self.result)