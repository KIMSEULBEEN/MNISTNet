import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, image = capture.read()
    cv2.imshow("image", image)
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'): break

capture.release()
cv2.destroyAllWindows()
