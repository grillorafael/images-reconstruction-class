def getFolder():
    return 'uff'

def getImages():
    return [
        'images/' + getFolder() +'/1.jpg',
        'images/' + getFolder() + '/2.jpg',
        'images/' + getFolder() + '/3.jpg'
    ]

def getPoints(np):
    points = {
        1: [
            [[663, 220], [763, 252], [664, 267], [764, 289]],
            [[173, 206], [280, 247], [174, 255], [280, 284]],
            None # matrix
        ],
        2: [
            [[900, 164], [1006, 200], [900, 216], [1013, 241]],
            [[285, 176], [379, 219], [285, 224], [385, 255]],
            None # matrix
        ],
        3: [
            [[873, 106], [1022, 84], [835, 544], [971, 620]],
            [[419, 83], [543, 64], [428, 508], [542, 555]],
            None # matrix
        ]
    }

    for index in points:
        points[index][0] = points[index][0][0:np]
        points[index][1] = points[index][1][0:np]

    return points
