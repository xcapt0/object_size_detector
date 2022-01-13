import cv2

from size_detector import SizeDetector


def launch_detector():
    print('Press "q" to quit the program')

    detector = SizeDetector()
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        img = detector.detect_contours(img)
        img = detector.draw_sizes(img)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)

        if key == 113 or key == 233:
            break


if __name__ == '__main__':
    launch_detector()
