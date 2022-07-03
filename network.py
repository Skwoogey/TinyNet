import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from ghost_module3d import GhostModule3D
from CenterLossLayer import CenterLossLayer
from ContrastiveCenterLossLayer import ContrastiveCenterLossLayer


def makeFast3DCNN():
    input_layer = tf.keras.Input(shape = (11, 11, 20, 1))
    num_of_classes = 16
    net = input_layer
    ghost_ratio = 2

    def localConv3D(inputs, filters, kernel_size):
        output = GhostModule3D(
            input=inputs,
            filters=filters,
            kernel_size=kernel_size,
            dw_kernel_size=kernel_size,
            padding='valid',
            ratio=ghost_ratio,
            stride=1,
            use_bias = False,
            use_BN=False,
            use_ReLU=True,
        )
        #output = tf.keras.layers.BatchNormalization()
        return output

    #shape (11, 11, 20, 1)
    net = tf.keras.layers.Conv3D(
            filters=8,
            kernel_size=(3, 3, 7),
            padding='valid',
            data_format='channels_last',
            activation='relu'
        )(net)
    #shape (9, 9, 14, 8)
    net = localConv3D(net, 16, (3, 3, 5))
    #shape (7, 7, 10, 16)
    net = localConv3D(net, 32, (3, 3, 3))
    #shape (5, 5, 8, 32)
    net = localConv3D(net, 64, (3, 3, 3))
    #shape (3, 3, 6, 64)
    net = tf.keras.layers.Flatten()(net)
    #shape (3456)
    net = tf.keras.layers.Dense(units = 256, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    #shape (256)
    net = tf.keras.layers.Dense(units = 128, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.4)(net)
    #shape (128)
    net = tf.keras.layers.Dense(units = num_of_classes, activation='softmax')(net)

    model = tf.keras.Model(inputs=input_layer, outputs=net)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return (model, optimizer)

def zero_loss(y_true, y_pred):
    return 0.5 * tf.keras.backend.sum(y_pred, axis=0)

def makeCustomFast3DCNN(
        input_window_width,
        input_window_depth,
        num_of_classes,
        conv_structure,
        # array of ('ghost' or 'conv', filters, (shape), ghost_ratio) elements
        dense_structure,
        # array of (units, activation, dropout_rate)
        ghost_ratio_override = None,
        use_BN = False,
        use_CenterLoss = False,
        CenterLoss_type ='CL',
        center_loss_weight = 1.0
    ):
    input_layer = tf.keras.Input(shape = (input_window_width, input_window_width, input_window_depth, 1))
    #input_layer = tf.keras.Input(shape = (input_window_width, input_window_width, 200))
    net = input_layer
    #net = tf.squeeze(net, -1)
    
    #net = tf.keras.layers.Conv2D(20, 1, 1, activation='relu')(net)
    #net = tf.keras.layers.BatchNormalization()(net)
    #net = tf.expand_dims(net, -1)


    input_labels_layer = None
    if use_CenterLoss:
        input_labels_layer = tf.keras.Input(shape = (num_of_classes,))

    
    def getGhostRatio(filters, gr):
        if ghost_ratio_override == None:
            return gr
        return min(filters, ghost_ratio_override)

    def localConv3D(inputs, filters, kernel_size, ghost_ratio, padding):
        output = GhostModule3D(
            input=inputs,
            filters=filters,
            kernel_size=kernel_size,
            dw_kernel_size=kernel_size,
            padding=padding,
            ratio=getGhostRatio(filters, ghost_ratio),
            stride=1,
            use_bias = True,
            use_BN=use_BN,
            use_ReLU=True,
        )
        #output = tf.keras.layers.BatchNormalization()
        return output

    for layer in conv_structure:
        if layer[0] == 'conv':
            net = tf.keras.layers.Conv3D(
            filters=layer[1],
            kernel_size=layer[2],
            padding=layer[4],
            data_format='channels_last',
            activation='relu'
            )(net)
        elif layer[0] == 'ghost':
            net = localConv3D(net,
                filters = layer[1],
                kernel_size = layer[2],
                ghost_ratio = layer[3],
                padding = layer[4]
            )
        if use_BN:
            net = tf.keras.layers.BatchNormalization()(net)

    net = tf.keras.layers.Flatten()(net)
    if dense_structure[0] != 0.0:
        net = tf.keras.layers.Dropout(dense_structure[0])(net)

    for layer in dense_structure[1:]:
        net = tf.keras.layers.Dense(units = layer[0], activation=layer[1])(net)
        if layer[2] != 0.0:
            net = tf.keras.layers.Dropout(layer[2])(net)
        if use_BN:
            net = tf.keras.layers.BatchNormalization()(net)

    output_loss_layer = None
    if use_CenterLoss:
        if CenterLoss_type == 'CL':
            CLL = CenterLossLayer
        elif CenterLoss_type == 'CCL':
            CLL = ContrastiveCenterLossLayer
        output_loss_layer = CLL(class_count = num_of_classes, name='cll')([net, input_labels_layer])


    net = tf.keras.layers.Dense(units = num_of_classes, activation='softmax', name='pr')(net)

    model = tf.keras.Model(inputs=input_layer, outputs=net)

    #optimizer = tfa.optimizers.AdamW(weight_decay=1e-5, amsgrad=False)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    optimizer = tf.keras.optimizers.SGD(momentum = 0.9, nesterov=False)
    #optimizer = tfa.optimizers.SGDW(weight_decay=1e-4, momentum = 0.9, nesterov=False)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model_cl = None
    if use_CenterLoss:
        model_cl = tf.keras.Model(inputs=[input_layer, input_labels_layer], outputs=[net, output_loss_layer])
        model_cl.compile(
            optimizer = optimizer,
            loss = [loss, zero_loss],
            loss_weights = [1, center_loss_weight],
            metrics = [tf.keras.metrics.CategoricalAccuracy()]
        )
    
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )

    return ((model, model_cl), optimizer)

