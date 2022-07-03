import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class ContrastiveCenterLossLayer(tf.keras.layers.Layer):

    def __init__(self, class_count, alpha=0.5, delta = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.class_count = class_count
        #self.ones = K.ones((1, self.class_count, 1), dtype=np.float32)
        self.ones = dict()
        self.ones[str(None)] = K.ones((1, self.class_count, 1), dtype=np.float32)
        self.delta = K.constant(delta)

    def build(self, input_shape):
        features_count = 1
        for i in range(1, len(input_shape[0])):
            features_count *= input_shape[0][i]

        self.centers = self.add_weight(name='centers',
                                       shape=(1, self.class_count, features_count),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is NxF, x[1] is NxC label, self.centers is 1xCxF

        #preparing elements
        FV = x[0]
        L = x[1]
        FV  = tf.expand_dims(FV, axis = -1)
        L = tf.expand_dims(L, axis = -1)
        centers_N = self.centers


        ### FORWARD PASS ###

        #print('FV:', FV.shape)
        #print('L:', L.shape)
        #print('CN:', centers_N.shape)

        # 1_c(N, C, 1) FV(N, F, 1)
        # 1_c * FV^T - (N, C, F)
        ones = None
        try:
            ones = self.ones[str(FV.shape[0])]
        except:
            self.ones[str(FV.shape[0])] = K.repeat_elements(self.ones[str(None)], FV.shape[0], axis=0)
            ones = self.ones[str(FV.shape[0])]
        #print(ones)

        FVV = K.batch_dot(ones, FV, (2, 2))
        
        #print('FVV:', FVV.shape)
        #(N, C, F)
        dFVV = FVV - centers_N
        #print('dFVV:', dFVV.shape)
        #(N, C, 1)
        dFVV2 = K.sum(dFVV ** 2, axis = 2, keepdims = True)
        #print('dFVV2:', dFVV2.shape)

        # L(N, C, 1) dFVV2(N, C, 1)
        # L^T * dFVV2 - (N, 1, 1)
        NUM = K.batch_dot(L, dFVV2, (1, 1))
        #print('NUM:', NUM.shape)

        #inverse_L (N, C, 1)
        inverse_L = ones - L

        # _L(N, C, 1) dFVV2(N, C, 1)
        # _L^T * dFVV2 - (N, 1, 1)
        DENUM = K.batch_dot(inverse_L, dFVV2, (1, 1))
        DENUM = DENUM + self.delta
        #print('DENUM:', DENUM.shape)
        

        #CCL - (N, 1, 1)
        CLPerSample = NUM / DENUM

        #CCL = (N, 1)
        CLPerSample = K.reshape(CLPerSample, (-1, 1))
        #print('res:', CLPerSample.shape)

        ### BACKWARD PASS ###

        #(N, 1, 1)
        DENUM2 = DENUM ** 2
        
        # -(N, C, F) * (N, C, 1) = (N, C, F)
        true_update_part = -dFVV * L
        #(N, C, F) /= (N, 1, 1)
        true_update_part /= DENUM
        #print('TUP:', true_update_part.shape)

        # (N, C, F) * (N, 1, 1) * (N, C, 1) = (N, C, F)
        false_update_part = dFVV * NUM * inverse_L
        #(N, C, F) /= (N, 1, 1)
        false_update_part /= DENUM2
        #print('FUP:', false_update_part.shape)
        #(N, C, F)
        update = true_update_part + false_update_part
        #(1, C, F)
        update = K.sum(update, axis = 0, keepdims = True)
        #print('U:', update.shape)
        new_centers = self.centers - self.alpha*update 
        #print('NC:', new_centers.shape)

        self.add_update((self.centers, new_centers))

        return CLPerSample

        ### END ###        

    def compute_output_shape(self, input_shape):
        return tf.keras.backend.int_shape(self.result)