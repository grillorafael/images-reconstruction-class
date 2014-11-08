import numpy as np
import timeit, sys, json
from cv2 import *

def main(args):
    imgs = args['images']
    # Loading images
    for i, imgPath in enumerate(imgs):
        imgs[i] = imread(imgPath)

    print "Collection", args['name']
    print "Processing", len(imgs), "images"

    # Truncating points
    points = args['relations']
    for index, point in enumerate(points):
        points[index][0] = points[index][0][0:args['np']]
        points[index][1] = points[index][1][0:args['np']]

    imageBuffer = result = np.zeros((10000, 20000, 3), np.uint8)

    for i, img in enumerate(imgs):
        print "\nProcessing image", i + 1
        h, w, c = img.shape
        if(i == 0):
            # Adding first image to the centerish of the image output
            for row in range(0, h):
                for column in range(0, w):
                    imageBuffer[row + 2000][column + 2000] = img[row][column]
        else:
            matrix = getHomography(args['method'], points[i - 1][1], points[i - 1][0], imgs[i - 1], img)
            if(i >= 2):
                matrix = points[i - 2][2].dot(matrix)
            points[i - 1][2] = matrix
            inverseMatrix = np.linalg.inv(matrix)

            # Bound of the new image
            p00 = matrix.dot([[0], [0], [1]])
            p00 = p00 / p00[2][0]

            p01 = matrix.dot([[0], [h - 1], [1]])
            p01 = p01 / p01[2][0]

            p10 = matrix.dot([[w - 1], [0], [1]])
            p10 = p10 / p10[2][0]

            p11 = matrix.dot([[w - 1], [h - 1], [1]])
            p11 = p11 / p11[2][0]

            listX = [int(p00[0][0]), int(p01[0][0]), int(p11[0][0]), int(p10[0][0])]
            listY = [int(p00[1][0]), int(p01[1][0]), int(p11[1][0]), int(p10[1][0])]

            minX, maxX = np.min(listX), np.max(listX)
            minY, maxY = np.min(listY), np.max(listY)

            for column in range(minX, maxX):
                for row in range(minY, maxY):
                    pt = [[column], [row], [1]]

                    transformPoint = inverseMatrix.dot(pt)
                    transformPoint = transformPoint / transformPoint[2][0]

                    height, width, channels = img.shape
                    if (width > transformPoint[0][0] >= 0) and (height > transformPoint[1][0] >= 0):
                            imageBuffer[pt[1][0] + 2000][pt[0][0] + 2000] = img[transformPoint[1][0]][transformPoint[0][0]]

    pathToJoin = args['output'].split('/')
    fileName = str(len(imgs)) + "_" + args['method'] + "_" + str(args['np']) + "_" + pathToJoin.pop()
    pathToJoin.append(fileName)
    saveDir = "/".join(pathToJoin)

    print "Saving to", saveDir
    imwrite(saveDir, imageBuffer)

def getHomography(method, p1, p2, img1, img2):
    if(method == 'dlt'):
        return dlt(p1, p2, img1, img2)
    elif(method == 'ndlt'):
        return dltNorm(p1, p2, img1, img2)
    elif(method == 'gold'):
        return gold(p1, p2, img1, img2)
    else:
        return dltNorm(p1, p2, img1, img2)

def gold(p1, p2, img1, img2):
    # TODO IMPLEMENT GOLD STANDARD
    return dltNorm(p1, p2, img1, img2)

def dlt(p1, p2, img1, img2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    print A
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    print "\n\n\n\n\n"
    print U
    print "\n\n\n\n\n"
    print S
    print "\n\n\n\n\n"
    print Vh
    print "\n\n\n\n\n"
    print H

    return H

def dltNorm(p1, p2, img1, img2):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    t = np.asarray([
        [h1 + w1, 0, w1/2],
        [0, h1 + w1, h1/2],
        [0, 0, 1]
    ])
    it = np.linalg.inv(t)


    tp = np.asarray([
        [h2 + w2, 0, w2/2],
        [0, h2 + w2, h2/2],
        [0, 0, 1]
    ])
    itp = np.linalg.inv(tp)

    tmpP1 = []
    tmpP2 = []

    for i in range(0, len(p1)):
        p1Tmp = p1[i]
        p1Tmp.append(1)

        p2Tmp = p2[i]
        p2Tmp.append(1)

        tmpP1.append(it.dot(p1[i]))
        tmpP2.append(itp.dot(p2[i]))

    hl = dlt(tmpP1, tmpP2, img1, img2)
    H = t.dot(hl).dot(itp)

    return H

def argsProcess(args):
    if '-file' in args:
        fileIndex = args.index('-file')
        fileName = args[fileIndex + 1]
        return json.load(open(fileName))
    else:
        print "Please pass the file parameter. Ex: -file filename.json"
        sys.exit()

start = timeit.default_timer()
main(argsProcess(sys.argv))
stop = timeit.default_timer()

print (stop - start), 's'
