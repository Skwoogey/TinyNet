import tensorflow as tf
import numpy as np
from network import makeFast3DCNN
from utils import createTrainDataPerClass, loadData, padImage, createTrainValidTestSplits
import sys
import os


# arg1 - image_file
# arg2 - ground_truth_file
# arg3 - size of the window
# arg4 - path to save model

data_file = sys.argv[1]#'.\\IPD\\PCA\\Indian_pines_20bands.mat'
gt_file = sys.argv[2]#'.\\IPD\\Indian_pines_gt.mat'
window_size = int(sys.argv[3])
model_base_path = sys.argv[4]

model, optimizer = makeFast3DCNN()
model.summary()

image, ground_truth =loadData(data_file, gt_file)
print(image.shape, image.dtype)
bands = image.shape[2]

padded_image = padImage(image, window_size)
print(padded_image.shape)

data_per_class = createTrainDataPerClass(padded_image, ground_truth, window_size)

train, valid, test = createTrainValidTestSplits(data_per_class, 0.35, 0.35, 0.3)

print((*train[0].shape, 1))
train[0] = train[0].reshape((*train[0].shape, 1))
valid[0] = valid[0].reshape((*valid[0].shape, 1))
test[0] = test[0].reshape((*test[0].shape, 1))

print("Train split", train[0].shape, train[1].shape)
print("Valid split", valid[0].shape, valid[1].shape)
print("Test split", test[0].shape, test[1].shape)

history = model.fit(
	x=train[0],
	y=train[1],
	batch_size=256,
	epochs=50,
	validation_data=(valid[0], valid[1]),
)

metrics = model.evaluate(test[0], test[1])

#print(history.history)
model.save(model_base_path + os.path.sep + 'model_' + str(metrics[1]) + '.h5')
