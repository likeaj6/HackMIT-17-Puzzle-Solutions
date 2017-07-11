import matplotlib.pyplot as plt
import os
import cv2
import json

category = []

def main():
    plt.ion()

    img_ext = '.jpg' #for example
    dirpath = './clean'
    img_name = [ x for x in os.listdir( dirpath ) if x.endswith(img_ext) ]
    img_fnames = [ os.path.join(dirpath, x) for x in os.listdir( dirpath ) if x.endswith(img_ext) ]

    allImages = []

    for i in range(len(img_fnames)):
        name = img_fnames[i]
        image = cv2.imread(name)
        allImages.append(image)
    with open('solution.json', 'w') as outfile:
        try:
            for i,image in enumerate(allImages):
                plt.figure(figsize=(2,1))
                plt.imshow(image)
                plt.pause(0.05)
                entry = {'name': img_name[i], 'solution': raw_input()}
                print(entry)
                category.append(entry)
        except KeyboardInterrupt:
             print(category)
             json.dump(category, outfile)
if __name__ == "__main__":
      main()
