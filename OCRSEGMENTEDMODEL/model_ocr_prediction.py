from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from textSegmentation import textSegment
from keras import backend as K
import cv2
from keras import utils
import numpy as np
import scipy.misc
import json
from PIL import Image
import os
import time

def int_to_label(index):
    characters = "abcdefghijklmnopqrstuvwxyz1234567890"
    letter = characters[index]
    return letter

num_classes = 36


loaded_model = Sequential()
loaded_model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 1)))
loaded_model.add(Activation('relu'))

loaded_model.add(Conv2D(128, (3, 3)))
loaded_model.add(Activation('relu'))

loaded_model.add(MaxPooling2D(pool_size=(2, 2)))

loaded_model.add(Conv2D(128, (3, 3)))
loaded_model.add(Activation('relu'))

loaded_model.add(Conv2D(128, (3, 3)))
loaded_model.add(Activation('relu'))

loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
loaded_model.add(Dropout(0.4))

loaded_model.add(Conv2D(128, (3, 3)))
loaded_model.add(Activation('relu'))

loaded_model.add(Conv2D(128, (3, 3)))
loaded_model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

loaded_model.add(Dropout(0.4))

loaded_model.add(Flatten())
loaded_model.add(Dense(5120))
# model.add(Activation('relu'))
loaded_model.add(Dropout(0.5))
loaded_model.add(Dense(num_classes))
loaded_model.add(Activation('softmax'))

loaded_model.summary()

loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# load weights into new model
loaded_model.load_weights("model.hdf5")
print("Loaded model from disk")
img_width = 32
img_height = 32


img_ext = '.jpg' #for example
dirpath = './clean/'
img_fnames = [ os.path.join(dirpath, x) for x in os.listdir( dirpath ) if x.endswith(img_ext) ]
img_name = [ x for x in os.listdir( dirpath ) if x.endswith(img_ext) ]

solved = []
total_solved = 0

with open('solution.json', 'w') as outfile:
    for i in range(len(img_fnames)):
        name = img_fnames[i]
        images = textSegment(name)
        if len(images) != 4:
            continue
        if total_solved == 9700:
            break
        print("SOLVED " + str(total_solved))
        print(str(len(images)) + " images")
        string = ""
        for img in images:
            arr = np.asarray(img).reshape(1, 32, 32, 1)
            prediction = loaded_model.predict(arr)
            result = prediction[0]
            print(result.max())
            out = np.argmax(result)
            char = int_to_label(out)
            string += char
        print(string)
        entry = {'name': img_name[i], 'solution': string}
        print(entry)
        solved.append(entry)
        total_solved += 1
    print("Solved " + str(total_solved) + " CAPTCHA!")
    print(solved)
    json.dump(solved, outfile)
