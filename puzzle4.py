from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
import cv2
from keras import utils
import numpy as np
import matplotlib
import scipy.misc
import PIL
from PIL import Image
import os
import time

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.hdf5")
print("Loaded model from disk")
img_width = 32
img_height = 32

loaded_model.summary()

# this is the placeholder for the input images
input_img = loaded_model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

layer_dict = dict([(layer.name, layer) for layer in loaded_model.layers[1:]])

layer_output = layer_dict['predictions'].output


loss = K.mean(loaded_model.output[:, 1])
# we compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads = normalize(grads)

# this function returns the loss and grads given the input picture

iterate = K.function([input_img], [loss, grads])

step = 0.3

np.random.seed(42)

# we start from a gray image with some random noise
if K.image_data_format() == 'channels_first':
    input_img_data = np.random.random((1, 3, img_width, img_height))
else:
    input_img_data = np.random.random((1, img_width, img_height, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128

prediction = loaded_model.predict(input_img_data)

y_proba = model.predict(input_img_data)
y_classes = keras.np_utils.probas_to_classes(prediction)
print(y_classes)

while (prediction[:,1].astype('float32') < 0.999):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

    print('Current loss value:', loss_value)
    scipy.misc.imsave('winner.jpg', input_img_data[0])
    img = Image.open("winner.jpg")
    img.load()
    data = np.asarray(img, dtype="uint8").reshape(1, 32, 32, 3)
    prediction = loaded_model.predict(data)
    print(prediction)
    if loss_value <= 0.:
        # some filters get stuck to 0, we can skip them
        break
