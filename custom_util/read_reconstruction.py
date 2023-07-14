import os
import json
import numpy as np

import sparsepointcloud

#https://opensfm.readthedocs.io/en/latest/dataset.html?highlight=sparse#reconstruction-file-format

reconstruction_path = '/media/mdai/Data_House_3/projects/ODM/projects/ikea_area/opensfm/reconstruction.json'

print('Reading reconstruciton file...')
with open(reconstruction_path, 'r') as f:
    data = json.load(f)

points = []
colors = []
for point in data[0]['points'].values():
    points.append(point['coordinates'])
    colors.append(point['color'])

filepath = './test.ply'

print('Exporting ply...')
sparsepointcloud.create_ply(filepath, points, colors)