import numpy as np
from cv2 import *

def main():
    img = imread('images/projective.jpg')

    height, width, ch = img.shape

    pts1 = np.float32([ [431, 110], [653, 48], [421, 561], [631, 655] ])
    pts2 = np.float32([ [0,0], [420,0], [0,607], [420,607] ])

    M = getPerspectiveTransformMatrix(pts1, pts2)

    dst = warpPerspective(img, M, (420, 607))

    imshow('image', dst)
    waitKey(0)
    destroyAllWindows()

def getPerspectiveTransformMatrix(p1, p2):
    return getPerspectiveTransform(p1, p2)

main()
