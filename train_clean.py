import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_images(name):
    folders = {"A", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
    images = []
    for folder in folders:
        findInFolder = "output/notMNIST_large/" + folder + "/" + name
        images.append(np.asarray(Image.open(findInFolder).convert('RGB')))
    return np.asarray(images)

def show_image(path):
    image = Image.open(path)
    Image._show(image)

def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity)).swapaxes(1,2).reshape((height*nrows, width*ncols, intensity)))
    return result

def make_array(png_path):
    pics = recognize(png_path)
    print("image size:", len(pics))
    images = []
    for pic in pics:
        images.append(np.asarray(Image.open(pic).convert('RGB')))
    return np.asarray(images)


def recognize(png_path, max = 1000):
    image_files = os.listdir(png_path)
    folders = {"A","b","c","d","e","f","g","h","i","j"}
    #folders = {"f"}
    images = []

    errorImages = ['RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png',
                   'SG90IE11c3RhcmQgQlROIFBvc3Rlci50dGY=.png',
                   'Um9tYW5hIEJvbGQucGZi.png'
                   ]

    for image in image_files:
        name = str(image)

        try:
            aaa =  errorImages.index(name)
            print(aaa,name)
            continue
        except:
            if name.endswith(".png"):
                #print(name)
                for folder in folders:
                    try:
                        findInFolder = "output/notMNIST_large/" + folder + "/" + name
                        images.append(findInFolder)
                        if len(images) == (max):
                            return images
                    except IOError as e:
                        print('Could not read:', e)


    return images


def show_filtered_dir():
    '''
    输入一个文件夹名。将文件夹内的文件做为查找字符串。去查找a-j内相同文件名的文件。并显示
    :return:
    '''
    array = make_array("output/notMNIST_large/A11")
    result = gallery(array)
    plt.imshow(result)
    plt.show()

#show_filtered_dir()


def showImageInAtoJByApath():
    '''
    输入一个文件的文件名，并在a-j文件夹内找到这个文件显示
    :return:
    '''
    result = gallery(show_images("RGV2aWwgQm9sZC50dGY=.png"))
    plt.imshow(result)
    plt.show()

#showImageInAtoJByApath()


def delFiles(files):
    for file in files:
        delFile(file=file)

def delFile(file):
    if os.path.exists(file):
        os.remove(file)
    else:
        print('no such file:%s' % file)

#delFile("output/notMNIST_large/A12/a2Fua2FuYSBLLnR0Zg==.png")

def delFileByIndexFolder(indexFolder):
    '''
    indexFolder做为"索引文件夹"，该文件夹内的所有文件作为要删除的文件。
    indexFolder文件夹的每一个文件，在a-j文件夹内都有对应文件。原因是
    它们都属于同一类字体。同类字体容易发生在表达字母a-j存在相同的缺陷，至少人不能理解其为字母
    所以该任务将移除这类"错误"(有些图片只是表达了字母意思。比如数字1-10对应字母a-j。单显然数字1就是1。我们不去猜想非视觉意外的内涵)的字体
    :param indexFolder:
    :return:
    '''
    pics = recognize(indexFolder, 65535)
    print("file size:(%s)" % len(pics))#18340
    delFiles(pics)


#delFileByIndexFolder("output/notMNIST_large/A11")


