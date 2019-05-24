from data_extraction import getdata
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Reference: https://www.kaggle.com/marwaf/handwritten-digits-classification-using-knn


# importing training and test features (i.e. images). The getdata() function
# returns only the digit that occupies the largest space or bounding square.
# Each element is a 2D array of size image_size X image_size
X_train = getdata('train', threshold=220, image_size=28)
X_test = getdata('test', threshold=220, image_size=28)

# normalizing data
X_train = X_train/255
X_test = X_test/255

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))
msamples, mx, my = X_test.shape
X_test = X_test.reshape((msamples,mx*my))

# importing training labels
y_train = pd.read_csv('train_labels.csv')['Category']
y_train = np.asarray(y_train)

# splitting X_train and y_train into training and validation sets
random_seed = 123
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=random_seed)

kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier

for k in range(1, 14, 2):
    print(k)
    # train the k-Nearest Neighbor classifier with the current value of
    # `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    # evaluate the model and update the accuracies list
    score = model.score(X_val, y_val)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy

i = np.argmax(accuracies)
print(
    "k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                     accuracies[
                                                                         i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(X_train, y_train)
predictions = model.predict(X_val)
# print(predictions[1])
#
# # show a final classification report demonstrating the accuracy of the classifier
# # for each of the digits
#
# print("EVALUATION ON TESTING DATA")
# print(classification_report(y_val, predictions))
#
# print("Confusion matrix")
# print(confusion_matrix(y_val, predictions))
#
# # loop over a few random digits
#
# for i in np.random.randint(0, high=len(y_val), size=(5,)):
#     # grab the image and classify it
#     image = X_val[i]
#     prediction = model.predict([image])[0]
#     # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
#     # then resize it to 32 x 32 pixels so we can see it better
#     ##         image = image.reshape((64, 64))
#     ##         image = exposure.rescale_intensity(image, out_range=(0, 255))
#     ##         image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
#
#     # show the prediction
#
#     imgdata = np.array(image, dtype='float')
#     pixels = imgdata.reshape((8, 8))
#     plt.imshow(pixels, cmap='gray')
#     plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
#     print("i think that digit is : {}".format(prediction))
#     # cv2.imshow("image", image)
#     plt.show()
#     cv2.waitKey(0)
#
# #
# # examples = enumerate(test_loader)
# # batch_idx, (example_data) = next(examples)
# #
# # # examples = enumerate(test_loader)
# # # batch_idx, (example_data, example_targets) = next(examples)
# #
# # with torch.no_grad():
# #     output = network(X_test_t)
# #
# # predict = np.zeros(len(example_data))
# #
# # for i in range(len(example_data)):
# #     predict[i] = output.data.max(1, keepdim=True)[1][i].item()
# #
# # # Predicting categories on test set and saving results as csv, ready for Kaggle
# # #my_prediction = lr.predict(X_combined_test)
# # my_prediction = np.array(predict).astype(int)
# # Id = np.linspace(0,9999, 10000).astype(int)
# # my_solution = pd.DataFrame(my_prediction, Id, columns = ['Category'])
# #
# # # Write Solution to a csv file with the name my_solution.csv
# my_solution.to_csv('my_solution.csv', index_label=['Id'])