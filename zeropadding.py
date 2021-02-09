import cv2

image = cv2.imread('sample.jpg', 0)
image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT)

cv2.imshow("img", image)
cv2.waitKey()
