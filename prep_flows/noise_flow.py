import numpy as np
import tensorflow as tf

def getFlow(train_x, train_y, args_batch_size, num_classes, prepDataFunc):
    
    #ones = np.identity(num_classes, dtype=np.float32)
    #train_y = [ones[x] for x in train_y]
    #print(train_y)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(args_batch_size * 100)
        .batch(args_batch_size)
    )

    def gaussian_noise(size, stddev):
        noise = tf.random.normal(shape=size, stddev=stddev)
        return noise

    def mix_up(x, y, stddev=0.1):
        # Unpack two datasets
        
        print("x, y")
        print(x, y)
        # Sample lambda and reshape it to do the mixup
        noise = gaussian_noise(tf.shape(x), stddev)

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        #print(labels_one)
        #print(y_l)
        
        return prepDataFunc(x + noise, y, num_classes)

    train_ds_mu = train_ds.map(
        lambda x, y: mix_up(x, y, stddev=0.05), num_parallel_calls=tf.data.AUTOTUNE
    )

    return train_ds_mu
