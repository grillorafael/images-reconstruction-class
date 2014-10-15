import numpy as np
import timeit
from cv2 import *

IMAGES = [
    imread('stereo/im0.png', 0),
    imread('stereo/im1.png', 0)
]

WINDOW_SIZE = 3

def main():
    print "Resizing images..."
    IMAGES[0] = addWindowFrame(IMAGES[0])
    IMAGES[1] = addWindowFrame(IMAGES[1])
    print "Finished Images Frame Resize..."

    height, width = IMAGES[0].shape
    outputImage = np.zeros((height, width), np.uint8)

    print "Calculating disparity..."
    for row in range(WINDOW_SIZE, height):
        progress = row / height
        print progress
        for column in range(WINDOW_SIZE, width):
            bestPosition = getDisparity((column, row))
            currentPosition = (column, row)
            disparity = distanceBetween(currentPosition, bestPosition)
            print disparity
            outputImage[row, column] = disparity

    print "Normalizing Output..."

    outputImage = np.asarray(outputImage)
    minElement = np.amin(outputImage)
    maxElement = np.amax(outputImage)

    outputImage = outputImage / maxElement
    outputImage = outputImage * 255

    imshow('image', outputImage)
    waitKey(0)
    destroyAllWindows()

def distanceBetween(point1, point2):
    x1 = point1[0]
    x2 = point2[0]

    y1 = point1[1]
    y2 = point2[1]

    result = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return result[0][0]

def getDisparity((x, y)):
    height, width = IMAGES[0].shape
    values = []
    positions = []
    for column in range(WINDOW_SIZE, width - WINDOW_SIZE):
        value = getValue('ssd', (column, y))
        values.append(value)
        positions.append((column, y))

    minPosition = values.index(np.amin(values))
    return positions[minPosition]

def getValue(method, (x, y)):
    if(method == 'ssd'):
        return ssd((x, y))

def ssd((x, y)):
    value = 0
    for row in range(y, y + WINDOW_SIZE):
        for column in range(x, x + WINDOW_SIZE):
            value += (IMAGES[0][row, column] - IMAGES[1][row, column]) ** 2
    return value

def addWindowFrame(image):
    height, width = image.shape
    newHeight = height + 2 * WINDOW_SIZE
    newWidth = width + 2 * WINDOW_SIZE

    outputImage = np.zeros((newHeight, newWidth), dtype='int64')

    for row in range(0, height):
        for column in range(0, width):
            outputImage[row + WINDOW_SIZE, column + WINDOW_SIZE] = image[row, column]

    return outputImage

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print (stop - start), 's'
