import itertools
import time

import cv2
import imutils as imutils
import numpy as np
from skimage.filters import threshold_local


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class Deskewing:

    def __init__(self, image, block_size=81, offset=10):
        self.image = image
        # coords is a tuple (x1, y1, x2, y2) representing the bounding
        # box of the detected grid
        self.coords = None

        self.block_size = block_size
        self.offset = offset

    def plot(self, plt):
        start = time.time()
        plt.imshow(self._deskew(), cmap='gray')
        time_used = time.time() - start

        plt.title(f'Detection Time: {time_used:.2}s')

    def _deskew(self):
        image = self.image
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        image = imutils.resize(image, height=500)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        # loop over the contours
        approx = []
        screenCnt = None

        # print('=')
        for c in cnts:
            # print(c.shape, cv2.contourArea(c))
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx.append(cv2.approxPolyDP(c, 0.02 * peri, True))

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            # print(approx[-1])
            if screenCnt is None and len(approx[-1]) == 4:
                screenCnt = approx[-1]

        # print(screenCnt)
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

        # apply the four point transform to obtain a top-down
        # view of the original image
        warped: np.array = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, self.block_size, offset=self.offset, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
        return warped.astype("uint8")


def deskew_verify(img):
    x, y = img.shape
    if x < 2000 or y < 2000 or abs(x - y) > 100:
        return False
    return True


def split_into_cells(cropped, border_size=3):
    width, height = cropped.shape
    digit_width, digit_height = width // 9, height // 9
    digits = np.zeros((81, 28, 28))

    ln_space = np.linspace(0, digit_width - 1, 28).astype(int)
    xv, yv = np.meshgrid(ln_space, ln_space)

    for i, (idx, jdx) in enumerate(itertools.product(range(9), range(9))):
        big_num = cropped[digit_width * idx: digit_width * (idx + 1),
                  digit_height * jdx: digit_height * (jdx + 1)]

        digits[i, :, :] = big_num[yv, xv]
        # cutting off the border
        border_size = 21 // 7
        pix_sum = 0.4
        for j in range(border_size):
            if digits[i, j, :].sum() < (digit_width * 255) * pix_sum:
                digits[i, j, :] = 255

            if digits[i, :, j].sum() < (digit_width * 255) * pix_sum:
                digits[i, :, j] = 255

            if digits[i, -j, :].sum() < (digit_width * 255) * pix_sum:
                digits[i, -j, :] = 255

            if digits[i, :, -j].sum() < (digit_width * 255) * pix_sum:
                digits[i, :, -j] = 255

    return digits
