import cv2

image = cv2.imread('img/sample4.jpg', 0)
height, width = image.shape
block_size = width // 5 if (width // 5) % 2 == 1 else width // 5 + 1
print("block size: ", block_size)

image_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 30)
image_bin = cv2.bitwise_not(image_bin)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin)

x_max, y_max, w_max, h_max, area_max = 0, 0, 0, 0, 0
for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]
    if (width*height / 2 > area > area_max):
        x_max, y_max, w_max, h_max, area_max = x, y, w, h, area

    # cv2.rectangle(image, (x, y, w, h), 0)


aspect_ratio = width / height # 가로/세로 비율, 클 수록 가로가 길다

print(aspect_ratio)
image_num = image_bin[y_max:y_max + h_max, x_max:x_max + w_max]
image_num = cv2.copyMakeBorder(image_num, int(width * aspect_ratio / 5), int(width * aspect_ratio / 5)
                               , int(height * aspect_ratio / 5), int(height * aspect_ratio / 5), cv2.BORDER_CONSTANT)
height, width = image_num.shape

aspect_num = width - height
if (aspect_num > 0):
    image_num = cv2.copyMakeBorder(image_num, aspect_num // 2, aspect_num // 2, 0, 0, cv2.BORDER_CONSTANT)
else:
    image_num = cv2.copyMakeBorder(image_num, 0, 0, abs(aspect_num // 2), abs(aspect_num // 2), cv2.BORDER_CONSTANT)


cv2.imwrite("img/tmp.jpg", image_num)

cv2.imshow("img", image_num)
cv2.waitKey()
