from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import json
import zipfile

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

datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.4, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.4, # randomly shift images vertically (fraction of total height)
        horizontal_flip=False, # randomly flip images
        vertical_flip=False) # randomly flip images

dataset_path = '/output/segmented/'

zip_ref = zipfile.ZipFile("segmented.zip", 'r')
zip_ref.extractall('/output')
zip_ref.close()

train_labels_file = 'training_data.json'
test_labels_file = 'testing_data.json'
num_classes = 36
#
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(test_labels_file)
#


x_train = load_images_from_pathnames(train_filepaths)
y_train = np_utils.to_categorical(train_labels, num_classes)
x_test = load_images_from_pathnames(test_filepaths)
y_test = np_utils.to_categorical(test_labels, num_classes)

print(x_train.shape)

datagen.fit(x_train)

# datagen.flow(x_train, y_train)

# train_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
# train_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.string)



model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 1)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(5120))
# model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/output/weights.hdf5', verbose=1, save_best_only=True)


# for i in range(20):
for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=6300): # these are chunks of 32 samples
    # print(i)
    # if i == 0:
    # serialize weights to HDF5
    # model.save_weights("/output/model.h5")
        # loss = model.train_on_batch(X_batch, Y_batch)
    model.fit(X_batch, Y_batch, epochs=200, validation_split = 0.05, verbose = 1, callbacks=[checkpointer])
    # model.fit(X_batch, Y_batch, epochs = 15, validation_split = 0.15, verbose = 1, callbacks=[checkpointer])
    # score = model.evaluate(x_test, y_test, verbose=1)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    # i+=1
    # if score[1] >= 0.925:
        # break
#
#
model_json = model.to_json()
with open("/output/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/output/model.h5")
score = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
