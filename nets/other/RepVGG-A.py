import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np

window_size = 9
window_depth = -1
def getModel(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)
    
    def block(x, width, stride=1):
        conv3 = keras.layers.Conv2D(width, 3, stride, "same")(x)
        conv3 = keras.layers.BatchNormalization()(conv3)
        
        conv1 = keras.layers.Conv2D(width, 1, stride, "same")(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        
        to_add = [conv3, conv1]
        if stride == 1 and x.get_shape()[-1] == width:
            to_add += [keras.layers.BatchNormalization()(x)]
        
        return keras.layers.Add()(to_add)
    
    def stage(x, width, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            x = block(x, width, stride)
            x = keras.layers.Activation("relu")(x)
            
        return x
            
        
    out = input_layer
    
    out = stage(out, 64, 5, 1)
    out = stage(out, 128, 4, 2)
    out = stage(out, 256, 2, 1)
    
    out = keras.layers.GlobalAveragePooling2D()(out)
    
    out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Dense(num_classes, activation='softmax')(out)
    
    model = tf.keras.Model(inputs=input_layer, outputs=out)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model, model
    
def prepareData(data, labels, num_classes):
    inputs = [data]
    outputs = [labels]
    return inputs, outputs
    
    
total_epochs = 240
def scheduler(epoch):
    lr = 0.01
    print(lr)
    return lr