import cv2
import cvlib as cv

img = cv2.imread('sample.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 찾기
faces, confidences = cv.detect_face(img)

for (x, y, x2, y2), conf in zip(faces, confidences):
		# 확률 출력하기
    cv2.putText(img, str(conf), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
		# 얼굴위치 bbox 그리기
    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

# 영상 출력
cv2.imshow('image', img)

key = cv2.waitKey(0)
cv2.destroyAllWindows()