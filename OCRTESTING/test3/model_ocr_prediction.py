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


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
img_width = 32
img_height = 32

def load_images_from_pathnames(paths):
    length = len(paths)
    data = np.empty((length, 32, 32, 1), dtype="float32")
    for index, path in enumerate(paths):
        img = Image.open(dataset_path+path)
        arr = np.asarray(img).reshape(32, 32, 1)
        data[index, :, :, :] = arr
    return data

def labels_to_ints(label):
    characters = "abcdefghijklmnopqrstuvwxyz1234567890"
    index = characters.index(str(label))
    return index

def int_to_label(index):
    characters = "abcdefghijklmnopqrstuvwxyz1234567890"
    letter = characters[index]
    return letter

def read_label_file(file):
    f = open(file, "r")
    json_file = json.load(f)
    filepaths = []
    labels = []
    for pic in json_file:
        name = pic['name']
        solution = pic['solution']
        filepaths.append(name)
        index = labels_to_ints(solution)
        labels.append(index)
    return filepaths, labels

test_labels_file = 'testing_data.json'
num_classes = 36

test_filepaths, test_labels = read_label_file(test_labels_file)
x_test = load_images_from_pathnames(test_filepaths)
y_test = np_utils.to_categorical(test_labels, num_classes)

img_ext = '.jpg' #for example
dirpath = './segmented/'
img_fnames = [ os.path.join(dirpath, x) for x in os.listdir( dirpath ) if x.endswith(img_ext) ]
img_name = [ x for x in os.listdir( dirpath ) if x.endswith(img_ext) ]

loaded_model.evaluate()
score = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# solved = []
# total_solved = 0
#
# with open('solution.json', 'w') as outfile:
#     for i in range(len(img_fnames)):
#         name = img_fnames[i]
#         images = textSegment(name)
#         if len(images) != 4:
#             continue
#         print(str(len(images)) + " images")
#         string = ""
#         for img in images:
#             arr = np.asarray(img).reshape(1, 32, 32, 1)
#             prediction = loaded_model.predict(arr)
#             result = prediction[0]
#             print(result.max())
#             out = np.argmax(result)
#             char = int_to_label(out)
#             string += char
#         print(string)
#         entry = {'name': img_name[i], 'solution': raw_input()}
#         print(entry)
#         solved.append(entry)
#         total_solved += 1
#     print("Solved" + str(total_solved) + " CAPTCHA!")
#     print(solved)
#     json.dump(solved, outfile)
