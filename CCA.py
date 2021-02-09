import cv2

image = cv2.imread('sample2.jpg', 0)
image_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 2)
image_bin = cv2.bitwise_not(image_bin)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin)


for i in range(1, cnt): 
    (x, y, w, h, area) = stats[i]
    if area < 50: continue

    cv2.rectangle(image, (x, y, w, h), 0)

cv2.imshow('src', image)
cv2.imshow('src_bin', image_bin)
cv2.waitKey()
cv2.destroyAllWindows()
