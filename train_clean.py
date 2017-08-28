import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def gallery(array, ncols=5):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity)).swapaxes(1,2).reshape((height*nrows, width*ncols, intensity)))
    return result

def make_array(png_path):
    pics = recognize(png_path)
    images = []
    for pic in pics:
        images.append(np.asarray(Image.open(pic).convert('RGB')))
    return np.asarray(images)


def recognize(png_path):
    image_files = os.listdir(png_path)
    # folders = {"A", "B", "C", "D", "E",
    #           "F", "G", "H", "I", "J",}

    folders = {"A","B"}
    images = []

    for image in image_files:
        name = str(image)
        print(name)

        for folder in folders:
            try:
                findInFolder = "output/notMNIST_large/"+folder+"/"+name
                images.append(findInFolder)
            except IOError as e:
                print('Could not read:',e)

    return images



array = make_array("output/notMNIST_large/A12",)
result = gallery(array)
plt.imshow(result)
plt.show()
