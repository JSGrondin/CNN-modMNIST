import pickle
from extract_digit import only_biggest_digit
import numpy as np


def getdata(dataset, threshold=220, image_size=28):
    # This function loads all images, then processes all images to return
    # only the digit that occupies that largest space (i.e. largest bounding
    # square)
    X=[]
    with open(dataset+'_images.pkl', 'rb') as f:
        images = pickle.load(f)
    for img in images:
        X.append(only_biggest_digit(img, threshold=threshold, size=image_size))
    X = np.asarray(X)
    return X

def get_whole_images(dataset):
    # This function loads all images, without thresholding and without
    # cropping.
    X=[]
    with open(dataset+'_images.pkl', 'rb') as f:
        images = pickle.load(f)
    for img in images:
        X.append(img)
    X = np.asarray(X)
    return X


