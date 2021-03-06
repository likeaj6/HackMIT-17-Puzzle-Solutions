import base64
import json
import cv2
import numpy as np
import scipy.misc

def greater(a, b):
    momA = cv2.moments(a)
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv2.moments(b)
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if xa > xb:
        return 1

    if xa == xb:
        return 0
    else:
        return -1

def smoothImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)

    return threshold


def cleanImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = smoothImage(img)
    or_img = cv2.bitwise_or(img, closing)
    return or_img


img_fnames = []
labels = []
data = []
directory = 'clean/'
with open('solution.json') as data_file:
    json_file = json.load(data_file)

    for pic in json_file:
        name = pic['name']
        solution = pic['solution']
        img_fnames.append(name)
        labels.append(solution)
    data_file.close()
for i in range(len(img_fnames)):
    name = directory + img_fnames[i] + '.jpg'
    image = cv2.imread(name)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
    _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold

    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

    # for each contour found, draw a rectangle around it on original image
    # print(contours)
    print(labels[i])
    contours.sort(greater)

    index = 0

    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        print(index)
        # discard areas that are too large
        if h>300 and w>300:
            continue
        # discard areas that are too small
        if h<10 or w<10:
            continue
        if index == 4:
            continue
        # draw rectangle around contour on original image
        # cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        if index < len(labels[i]):
            region = image[y: y + h, x: x + w]
        # write original image with added contours to disk

            region = cleanImage(region)
            # print(region.shape)
            img = cv2.resize(region, (32, 32))
            # print(img.shape)
            img_path = img_fnames[i]+str(index)+".jpg"
            scipy.misc.imsave("segmented/" + img_path,img)
            entry = {'name': img_path, 'solution': labels[i][index]}
            data.append(entry)
            index += 1
with open('training_data.json', 'w') as output_file:
    json.dump(data, output_file)
