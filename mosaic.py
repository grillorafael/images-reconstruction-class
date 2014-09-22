import numpy as np
from cv2 import *

HOMOGENEOUS_COORDINATE = [[0], [0], [1]];

def main():
    # imgs = ['images/1.jpg', 'images/2.jpg', 'images/3.jpg', 'images/4.jpg']
    imgs = ['images/1.jpg', 'images/2.jpg', 'images/3.jpg']
    for i, imgPath in enumerate(imgs):
        imgs[i] = imread(imgPath)

    print "Processing", len(imgs), "images"

    points = {
        1: [
            [[841, 214], [910, 214], [835, 297], [902, 300]],
            [[158, 188], [226, 189], [161, 273], [228, 275]],
            None # matrix
        ],
        2: [
            [[597, 149], [693, 150], [592, 274], [685, 276]],
            [[64, 114], [173, 122], [70, 253], [177, 255]],
            None # matrix
        ],
        3: [
            [[873, 106], [1022, 84], [835, 544], [971, 620]],
            [[419, 83], [543, 64], [428, 508], [542, 555]],
            None # matrix
        ]
    }

    for i, img in enumerate(imgs):
        print "Processing image", i + 1
        h, w, c = img.shape
        if(i == 0):
            resultImage = img
        else:
            matrix = getPerspectiveTransformMatrix(points[i][1], points[i][0])
            if(i >= 2):
                matrix = points[i - 1][2].dot(matrix)
                # matrix = matrix / matrix[2][2]
            points[i][2] = matrix
            inverseMatrix = np.linalg.inv(matrix)

            # Bound of the new image
            p00 = matrix.dot([[0], [0], [1]])
            p00 = p00 / p00[2][0]

            p01 = matrix.dot([[0], [767], [1]])
            p01 = p01 / p01[2][0]

            p10 = matrix.dot([[1023], [0], [1]])
            p10 = p10 / p10[2][0]

            p11 = matrix.dot([[1023], [767], [1]])
            p11 = p11 / p11[2][0]

            listX = [int(p00[0][0]), int(p01[0][0]), int(p11[0][0]), int(p10[0][0])]
            listY = [int(p00[1][0]), int(p01[1][0]), int(p11[1][0]), int(p10[1][0])]

            minX, maxX = np.min(listX), np.max(listX)
            minY, maxY = np.min(listY), np.max(listY)

            print "MinX", minX
            print "MaxX", maxX
            print "MinY", minY
            print "MaxY", maxY
            print "Check inverse", inverseMatrix.dot([[841], [214], [1]])

            resultImage = resize(resultImage, (maxX, maxY))

            for column in range(minX, maxX):
                # print "X", column
                for row in range(minY, maxY):
                    # print "Y", row

                    pt = [[column], [row], [1]]

                    transformPoint = inverseMatrix.dot(pt)
                    transformPoint = transformPoint / transformPoint[2][0]

                    # print "Pt", pt[0][0], pt[1][0]
                    # print "Tp", transformPoint[0][0], transformPoint[1][0]
                    height, width, channels = img.shape
                    if (width > transformPoint[0][0] >= 0) and (height > transformPoint[1][0] >= 0):
                        if (20000 > pt[0][0] >= 0) and (20000 > pt[1][0] >= 0):
                            resultImage[pt[1][0]][pt[0][0]] = img[transformPoint[1][0]][transformPoint[0][0]]

            # for column in range(0, w):
            #     for row in range(0, h):
            #         pt = [[column], [row], [1]]
            #
            #         transformPoint = matrix.dot(pt)
            #         transformPoint = transformPoint / transformPoint[2][0]
            #
            #         # print "Pt", pt[0][0], pt[1][0]
            #         # print "Tp", transformPoint[0][0], transformPoint[1][0]
            #         resultImage[transformPoint[1][0]][transformPoint[0][0]] = img[pt[1][0]][pt[0][0]]
    imwrite("mosaic.png", resultImage);
    imshow('image', resultImage)
    waitKey(0)
    destroyAllWindows()

def getPerspectiveTransformMatrix(p1, p2):
    # 2D
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)

    return H

def resize(matrix, size):
    beforeY, beforeX, beforeC = matrix.shape
    result = np.zeros((size[1], size[0], 3), np.uint8)

    print "before shape", matrix.shape
    print "New shape", result.shape

    # offsetX = (size[0] - beforeX) / 2
    offsetY = (size[1] - beforeY) / 2

    print "Resizing from", (beforeX, beforeY)
    print "Resizing to", size
    print "Offsets", (0, offsetY)

    for row in range(0, beforeY):
        for column in range(0, beforeX):
            # print "X,Y", column, row
            # print "Test", result[2155][940]
            result[row][column] = matrix[row][column]
            # result[row][column] = matrix[row][column]
            # result[row][(column + offsetY) - 1] = 1
            # result[row][column + offsetY] = matrix[row][column]

    return result

def concatenateImg(image1, image2, origin):
    height, width, channels = image2.shape
    print "Concatenating", height, width
    print "Origin", origin
    for column in range(0, width):
        for row in range(0, height):
            image1[row + origin[1], column + origin[0]] = image2[row, column]
    return image1

main()
