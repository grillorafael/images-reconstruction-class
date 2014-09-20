import numpy as np
from cv2 import *

HOMOGENEOUS_COORDINATE = [[0], [0], [1]];

def main():
    # images in the correct order
    # imgs = ['images/1.jpg', 'images/2.jpg', 'images/3.jpg', 'images/4.jpg']
    imgs = ['images/1.jpg', 'images/2.jpg']
    maxWidth = maxHeight = 0
    for i, imgPath in enumerate(imgs):
        imgs[i] = imread(imgPath)
        height, width, channels = imgs[i].shape
        if(width > maxWidth):
            maxWidth = width
        if(height > maxHeight):
            maxHeight = height

    maxWidth = maxWidth * (len(imgs) + 1)
    maxHeight = maxHeight * (len(imgs) + 1)
    print "maxWidth", maxWidth
    print "maxHeight", maxHeight
    print "Processing", len(imgs), "images"

    print "Creating output image.."
    resultImage = np.zeros((maxHeight, maxWidth, 3), np.uint8)

    points = {
        1: [
            np.float32([[841, 214], [909, 215], [836, 297], [904, 300]]),
            np.float32([[159, 188], [223, 191], [161, 272], [227, 275]]),
            None # matrix
        ],
        2: [
            np.float32([[598, 149], [692, 151], [592, 273], [684, 276]]),
            np.float32([[64, 115], [173, 122], [71, 251], [176, 255]]),
            None # matrix
        ],
        3: [
            np.float32([[873, 106], [1022, 84], [835, 544], [971, 620]]),
            np.float32([[419, 83], [543, 64], [428, 508], [542, 555]]),
            None # matrix
        ]
    }

    for i, img in enumerate(imgs):
        print "Processing image", i + 1
        h, w, c = img.shape
        if(i == 0):
            resultImage = concatenateImg(resultImage, img, [0, 0])
        else:
            matrix = getPerspectiveTransformMatrix(points[i][1], points[i][0])
            # pt = matrix.dot(HOMOGENEOUS_COORDINATE)
            # print pt
            pt = matrix.dot([[159], [188], [1]])
            print "First point map", pt
            # pt = matrix.dot([[599], [146], [1]])
            # print pt
            origin = matrix.dot(HOMOGENEOUS_COORDINATE)
            print "Origin", origin
            for column in range(0, w):
                for row in range(0, h):
                    pt = [[column], [row], [1]]
                    transformPoint = matrix.dot(pt)
                    transformPoint = transformPoint / transformPoint[2][0]

                    print "Pt", pt[0][0], pt[1][0]
                    print "Tp", transformPoint[0][0], transformPoint[1][0]
                    resultImage[transformPoint[1][0]][transformPoint[0][0]] = img[pt[1][0]][pt[0][0]]

            # pt = matrix.dot([[159], [188], [1]]) # where the point will be mapped to
            # pt = pt / pt[2]
            # print pt
            # print matrix.dot(HOMOGENEOUS_COORDINATE) # Image origin position
            # if i >= 2:
            #
            # print matrix
            # if(i >= 2):
            #     print "Multiplying matrix from image", i + 1, i
            #
            #     matrix = np.dot(points[i - 1][2], matrix)
            #     print "Not normalized", matrix
            #
            #     matrix = matrix / matrix[2][2]
            #     print "Normalized", matrix

            # points[i][2] = matrix;
            # rectifImg = applyTransformationMatrix(img, matrix, (w, h))
            # resultImage = concatenateImg(resultImage, rectifImg, [1024 * i, 0])

    imshow('image', resultImage)
    waitKey(0)
    destroyAllWindows()

def getPerspectiveTransformMatrix(p1, p2):
    # OpenCV Method
    # return getPerspectiveTransform(p1, p2)
    matrixIndex = 0
    A = np.zeros((8, 9))
    for i in range(0, len(p1)):
        x = p1[i][0]
        y = p1[i][1]

        u = p2[i][0]
        v = p2[i][1]

        A[matrixIndex] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        A[matrixIndex + 1] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]

        matrixIndex = matrixIndex + 2

    U, s, V = np.linalg.svd(A, full_matrices=True)
    matrix = V[:, 8].reshape(3, 3)
    # Normalization
    matrix = matrix / matrix[2][2]
    return matrix

def applyTransformationMatrix(image, matrix, outputSize):
    return warpPerspective(image, matrix, outputSize)

def concatenateImg(image1, image2, origin):
    height, width, channels = image2.shape
    print "Concatenating", height, width
    print "Origin", origin
    for column in range(0, width):
        for row in range(0, height):
            image1[row + origin[1], column + origin[0]] = image2[row, column]
    return image1

main()
