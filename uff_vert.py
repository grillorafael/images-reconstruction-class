def getFolder():
    return 'uff_vert'

def getImages():
    return [
        'images/' + getFolder() +'/1.jpg',
        'images/' + getFolder() + '/2.jpg',
        'images/' + getFolder() + '/3.jpg'
    ]

def getPoints(np):
    points = {
        1: [
            [[662, 104], [762, 158], [662, 151], [762, 195]],
            [[635, 460], [737, 507], [639, 502], [743, 542]],
            None # matrix
        ],
        2: [
            [[761, 602], [828, 620], [766, 640], [833, 652]],
            [[289, 251], [350, 274], [290, 287], [351, 303]],
            None # matrix
        ],
    }

    for index in points:
        points[index][0] = points[index][0][0:np]
        points[index][1] = points[index][1][0:np]

    return points
