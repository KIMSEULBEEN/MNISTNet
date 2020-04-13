import cv2
import numpy as np

num_train, num_test = 84000, 14000


train_array = []
for n in range(1,num_train+1):
    print(n)
    img = cv2.imread("train/{}.png".format(n), 0)
    img = np.ravel(img)
    train_array.append(img)
np.save("train.npy", train_array)

test_array = []
for n in range(1,num_test+1):
    print(n)
    img = cv2.imread("test/{}.png".format(n), 0)
    img = np.ravel(img)
    test_array.append(img)
np.save("test.npy", test_array)
