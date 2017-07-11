from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.datasets import cifar10
from textSegmentation import textSegment
from keras import backend as K
import cv2
from keras import utils
import numpy as np
import scipy.misc
from PIL import Image
import os
import time

def int_to_label(index):
    characters = "abcdefghijklmnopqrstuvwxyz1234567890"
    letter = characters[index]
    return letter

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
img_width = 32
img_height = 32

image = "3.jpg"
images = textSegment(image)
print(str(len(images)) + " images")
string = ""
for img in images:
    arr = np.asarray(img).reshape(1, 32, 32, 1)
    prediction = loaded_model.predict(arr)
    result = prediction[0]
    print(result.max())
    out = np.argmax(result)
    char = int_to_label(out)
    print(char)
    string += char
print(string)
