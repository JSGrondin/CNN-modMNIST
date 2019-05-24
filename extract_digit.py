import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import time
import imutils

# Simple function to allow user to press key to go to next iteration
import msvcrt as m
def wait():
    m.getch()

# importing training and test modified MNIST images
with open('train_images.pkl', 'rb') as f:
    train = pickle.load(f)
with open('test_images.pkl', 'rb') as f:
    test = pickle.load(f)


def show_digit(img, threshold=220):
    # This function will take the original image as an input and plot the
    # bounding box around the digit that occupies the largest space; as well
    # as the cropped and binarized image of the biggest digit.

    # clearing previous plots
    plt.clf()

    # converting image to int8
    img=img.astype(np.uint8)

    # REFERENCE
    # https://gist.github.com/bigsnarfduded811e31ee17495f82f10db12651ae82d

    # threshold image to binarize it
    ret, threshed_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    # for the largest contour, draw bounding rectangle
    side_max = 0
    index_max = 0

    # Identify the largest contour, i.e. the one with the largest bounding
    # square
    for idx, c in enumerate(contours):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w > side_max:
            index_max = idx
            side_max = w
        if h > side_max:
            index_max = idx
            side_max = h

    # draw a green square to visualize the largest bounding square - while
    # centering the digit in the center of the cube
    x, y, w, h = cv2.boundingRect(contours[index_max])
    if side_max == w:
        delta = (side_max - h) // 2
        if (y-delta+side_max) <= len(img):
            if (y-delta) >= 0:
                y= y-delta
            else:
                y=0
    else:
        delta = (side_max - w) // 2
        if (x - delta + side_max) <= len(img):
            if (x - delta) >= 0:
                x = x - delta
            else:
                x = 0

    # showing original image with square bounding box
    plt.subplot(1, 2, 2)
    #    plt.imshow(threshed_img[y:y+side_max, x:x+side_max])
    plt.imshow(img[y:y + side_max, x:x + side_max])

    cv2.rectangle(img, (x, y), (x + side_max, y + side_max), (0, 255, 0), 1)
    plt.subplot(1, 2, 1)
    plt.imshow(img)


    return


def show_resized_digit(img, threshold=220, size=32):
    # This function will take the original image as an input and plot the
    # bounding box around the digit that occupies the largest space; as well
    # as the cropped and binarized image of the biggest digit.

    # clearing previous plots
    plt.clf()

    # converting image to int8
    img=img.astype(np.uint8)

    # REFERENCE
    # https://gist.github.com/bigsnarfduded811e31ee17495f82f10db12651ae82d

    # threshold image to binarize it
    ret, threshed_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    # for the largest contour, draw bounding rectangle
    side_max = 0
    index_max = 0

    # Identify the largest contour, i.e. the one with the largest bounding
    # square
    for idx, c in enumerate(contours):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w > side_max:
            index_max = idx
            side_max = w
        if h > side_max:
            index_max = idx
            side_max = h

    # draw a green square to visualize the largest bounding square - while
    # centering the digit in the center of the cube
    x, y, w, h = cv2.boundingRect(contours[index_max])
    if side_max == w:
        delta = (side_max - h) // 2
        if (y-delta+side_max) <= len(img):
            if (y-delta) >= 0:
                y= y-delta
            else:
                y=0
    else:
        delta = (side_max - w) // 2
        if (x - delta + side_max) <= len(img):
            if (x - delta) >= 0:
                x = x - delta
            else:
                x = 0

    # showing original image with square bounding box
    cv2.rectangle(img.copy(), (x, y), (x + side_max, y + side_max), (0, 255,
                                                                  0), 1)
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    biggest_digit = img[y:y+side_max, x:x+side_max]
    resized_digit = cv2.resize(biggest_digit, (size, size),
                               interpolation=cv2.INTER_AREA)
    ret2, final_img = cv2.threshold(resized_digit, threshold, 255,
                                      cv2.THRESH_BINARY)

    plt.subplot(1, 2, 2)
    plt.imshow(resized_digit)
    return

def show_resized_digit2(img, threshold=220, size=32):
    # This function will take the original image as an input and plot the
    # bounding box around the digit that occupies the largest space; as well
    # as the cropped and binarized image of the biggest digit.

    # clearing previous plots
    plt.clf()

    # converting image to int8
    img=img.astype(np.uint8)

    # REFERENCE
    # https://gist.github.com/bigsnarfduded811e31ee17495f82f10db12651ae82d

    # threshold image to binarize it
    ret, threshed_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    # for the largest contour, draw bounding rectangle
    side_max = 0
    index_max = 0

    # Identify the largest contour, i.e. the one with the largest bounding
    # square
    for idx, c in enumerate(contours):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w > side_max:
            index_max = idx
            side_max = w
        if h > side_max:
            index_max = idx
            side_max = h

    # draw a green square to visualize the largest bounding square - while
    # centering the digit in the center of the cube
    x, y, w, h = cv2.boundingRect(contours[index_max])
    if side_max == w:
        delta = (side_max - h) // 2
        if (y-delta+side_max) <= len(img):
            if (y-delta) >= 0:
                y= y-delta
            else:
                y=0
    else:
        delta = (side_max - w) // 2
        if (x - delta + side_max) <= len(img):
            if (x - delta) >= 0:
                x = x - delta
            else:
                x = 0

    # showing original image with square bounding box
    cv2.rectangle(img.copy(), (x, y), (x + side_max, y + side_max), (0, 255,
                                                                  0), 1)
    plt.subplot(1, 3, 1)
    plt.imshow(img)

    biggest_digit = img[y:y+side_max, x:x+side_max]
    resized_digit = cv2.resize(biggest_digit, (size, size),
                               interpolation=cv2.INTER_AREA)
    ret2, final_img = cv2.threshold(resized_digit, threshold, 255,
                                      cv2.THRESH_BINARY)

    plt.subplot(1, 3, 2)
    plt.imshow(resized_digit)

    plt.subplot(1, 3, 3)
    plt.imshow(final_img)
    return


def only_biggest_digit(img, threshold=220, size=32):
    # This function will take the original image as an input and return a
    # croped image containing only the digit that occupied the largest space
    # in the original image. The output will be a numpy array.

    # converting image to int8
    img=img.astype(np.uint8)

    # REFERENCE
    # https://gist.github.com/bigsnarfduded811e31ee17495f82f10db12651ae82d

    # threshold image to binarize it
    ret, threshed_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    # for the largest contour, draw bounding rectangle
    side_max = 0
    index_max = 0

    # Identify the largest contour, i.e. the one with the largest bounding
    # square
    for idx, c in enumerate(contours):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if w > side_max:
            index_max = idx
            side_max = w
        if h > side_max:
            index_max = idx
            side_max = h

    # draw a green square to visualize the largest bounding square - while
    # centering the digit in the center of the cube
    x, y, w, h = cv2.boundingRect(contours[index_max])
    if side_max == w:
        delta = (side_max - h) // 2
        if (y-delta+side_max) <= len(img):
            if (y-delta) >= 0:
                y= y-delta
            else:
                y=0
    else:
        delta = (side_max - w) // 2
        if (x - delta + side_max) <= len(img):
            if (x - delta) >= 0:
                x = x - delta
            else:
                x = 0

    biggest_digit = img[y:y+side_max, x:x+side_max]
    resized_digit = cv2.resize(biggest_digit, (size, size),
                               interpolation=cv2.INTER_AREA)
    ret2, final_img = cv2.threshold(resized_digit, threshold, 255,
                                      cv2.THRESH_BINARY)

    return final_img

#
# #importing training labels
# y_train = pd.read_csv('train_labels.csv')['Category']
#
# plt.clf()
# plt.ion()
# plt.show()
# # testing the show_digit function
# for i in range(len(train)):
# #i = 0
#
#     #plt.clf()
#     #show_digit(train[i], 220)
#     # time.sleep(0.1)
#     # plt.pause(0.0001)
#     # input("Press Enter to continue...")
#     show_resized_digit2(train[i],220,28)
#     # plt.subplot(1, 3, 1)
#     # plt.title(''.join(["Original 64x64, label:", str(y_train[i])]))
#     # plt.subplot(1, 3, 2)
#     # plt.title(''.join(["Resized 28x28, non-binarized"]))
#     # plt.subplot(1, 3, 3)
#     # plt.title(''.join(["Resized 28x28, threshold (220):"]))
# #    plt.title(''.join(["idx:",str(i), " label:", str(y_train[i])]))
#     plt.draw()
#     time.sleep(0.1)
#     plt.pause(0.0001)
#     input("Press Enter to continue...")

# print('done')
#
# # testing the only_biggest_digit function
# big_img = only_biggest_digit(train[i], 220, 32)
# np.shape(big_img)