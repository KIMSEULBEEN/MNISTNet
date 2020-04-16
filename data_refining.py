import cv2
import numpy as np

num_train, num_test = 5, 14000

def npy_gen(rdir, num_images):
    data_array = []
    for n in range(1, num_images + 1):
        print(n)
        img = cv2.imread("{}/{}.png".format(rdir, n), 0)
        img = np.ravel(img)
        data_array.append(img)
    np.save("{}.npy".format(rdir), data_array)


npy_gen("train", num_train)
npy_gen("test", num_test)
