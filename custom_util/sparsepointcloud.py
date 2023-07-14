import numpy as np
from plyfile import PlyData, PlyElement

# Assume we have some data like this:
#points = np.array([
#    [1, 2, 3],
#    [4, 5, 6],
#    [7, 8, 9]
#], dtype=np.float32)
#
#
#colors = np.array([
#    [255, 0, 0],
#    [0, 255, 0],
#    [0, 0, 255]
#], dtype=np.uint8)

def create_ply(filepath, points, colors):
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    data = np.empty(points.shape[0], dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1')
    ])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['red'] = colors[:, 0]
    data['green'] = colors[:, 1]
    data['blue'] = colors[:, 2]

    element = PlyElement.describe(data, 'vertex')
    PlyData([element], text=True).write(filepath)