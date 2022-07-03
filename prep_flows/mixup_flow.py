import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def getFlow(train_x, train_y, args_batch_size, num_classes, prepDataFunc):
    
    #ones = np.identity(num_classes, dtype=np.float32)
    #train_y = [ones[x] for x in train_y]
    #print(train_y)

    train_ds_one = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(args_batch_size * 100)
        .batch(args_batch_size)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(args_batch_size * 100)
        .batch(args_batch_size)
    )
    # Because we will be mixing up the images and their corresponding labels, we will be
    # combining two shuffled datasets from the same training data.
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))


    alpha=0.2
    beta_sampler = tfp.distributions.Beta(alpha, alpha)
    def mix_up(ds_one, ds_two):
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]
        
        l = beta_sampler.sample(batch_size)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        #print(labels_one)
        #print(y_l)

        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)
        
        return prepDataFunc(images, labels, num_classes)

    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two), num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_ds_mu
