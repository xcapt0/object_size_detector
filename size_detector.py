import cv2
import numpy as np


class SizeDetector():
    def __init__(self):
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        self.contours = None
        self.ratio = None

    def detect_contours(self, img):
        self.corners, _, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)

        if self.corners:
            int_corners = np.int0(self.corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            self._calibrate_ratio()

            contours = self._get_contours(img)
            self.contours = contours
        return img

    def draw_sizes(self, img):
        if self.corners and self.ratio:
            for cnt in self.contours:
                rectangle = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rectangle

                box = np.int0(cv2.boxPoints(rectangle))
                cv2.polylines(img, [box], True, (255, 0, 0), 2)

                object_width = w / self.ratio
                object_height = h / self.ratio

                cv2.putText(
                    img,
                    f'Width: {round(object_width, 1)} cm',
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (255, 255, 0), 2
                )
                cv2.putText(
                    img,
                    f'Height: {round(object_height, 1)} cm',
                    (int(x), int(y + 25)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5, (255, 255, 0), 2
                )

            self._reset_contours()
        return img

    def _calibrate_ratio(self):
        calculated_perimeter = cv2.arcLength(self.corners[0], True)
        marker_width, marker_height = 5, 5
        self.ratio = calculated_perimeter / (2 * (marker_width + marker_height))

    def _reset_contours(self):
        self.contours = None
        self.ratio = None

    @staticmethod
    def _get_contours(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)

        return objects_contours
