#!/usr/bin/env python
"""
    File Name   :   s3g-vis_results
    date        :   3/12/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""
import plyfile
from os.path import join

import numpy as np
from glob import glob
import pdb

from tqdm import tqdm
import multiprocessing as mp


# color palette for nyu40 labels
def create_color_palette():
    return np.array([
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144)
    ])


def icosahedron():
    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    sphereLength = np.sqrt(PHI * PHI + 1.0)
    dist1 = PHI / sphereLength
    dist2 = 1.0 / sphereLength

    verts = [
        [-dist2, dist1, 0], [dist2, dist1, 0], [-dist2, -dist1, 0], [dist2, -dist1, 0],
        [0, -dist2, dist1], [0, dist2, dist1], [0, -dist2, -dist1], [0, dist2, -dist1],
        [dist1, 0, -dist2], [dist1, 0, dist2], [-dist1, 0, -dist2], [-dist1, 0, dist2]
    ]

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    return verts, faces


def createEdgeIndex(index1, index2, totalVerts):
    if index1 > index2:
        auxVal = index1
        index1 = index2
        index2 = auxVal
    index1 *= totalVerts
    outIndex = index1 + index2
    return outIndex


def subdivide(verts, faces):
    triangles = len(faces)
    edgeMap = dict([])
    currLength = len(verts)
    for faceIndex in range(triangles):
        face = faces[faceIndex]
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]

        v3EdgeIndex = createEdgeIndex(face[0], face[1], currLength)
        v3Index = -1
        if v3EdgeIndex in edgeMap:
            v3Index = edgeMap[v3EdgeIndex]
        else:
            newVert = np.array([(v0[0] + v1[0]) * 0.5, (v0[1] + v1[1]) * 0.5, (v0[2] + v1[2]) * 0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0] / length, newVert[1] / length, newVert[2] / length])
            edgeMap[v3EdgeIndex] = len(verts) - 1
            v3Index = len(verts) - 1

        v4EdgeIndex = createEdgeIndex(face[1], face[2], currLength)
        v4Index = -1
        if v4EdgeIndex in edgeMap:
            v4Index = edgeMap[v4EdgeIndex]
        else:
            newVert = np.array([(v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5, (v1[2] + v2[2]) * 0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0] / length, newVert[1] / length, newVert[2] / length])
            edgeMap[v4EdgeIndex] = len(verts) - 1
            v4Index = len(verts) - 1

        v5EdgeIndex = createEdgeIndex(face[0], face[2], currLength)
        v5Index = -1
        if v5EdgeIndex in edgeMap:
            v5Index = edgeMap[v5EdgeIndex]
        else:
            newVert = np.array([(v0[0] + v2[0]) * 0.5, (v0[1] + v2[1]) * 0.5, (v0[2] + v2[2]) * 0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0] / length, newVert[1] / length, newVert[2] / length])
            edgeMap[v5EdgeIndex] = len(verts) - 1
            v5Index = len(verts) - 1

        faces.append([v3Index, v4Index, v5Index])
        faces.append([face[0], v3Index, v5Index])
        faces.append([v3Index, face[1], v4Index])
        faces[faceIndex] = [v5Index, v4Index, face[2]]

    return verts, faces


def save_fancy_model_ply(modelName, points, labels, sphPts, sphFaces, sphScale):
    coordMax = np.amax(points, axis=0)
    coordMin = np.amin(points, axis=0)
    aabbSize = (1.0 / np.amax(coordMax - coordMin)) * sphScale

    newModelName = modelName[:-4] + "_spheres.ply"
    with open(newModelName, 'w') as myFile:
        myFile.write("ply\n")
        myFile.write("format ascii 1.0\n")
        myFile.write("element vertex " + str(len(sphPts) * len(points)) + "\n")
        myFile.write("property float x\n")
        myFile.write("property float y\n")
        myFile.write("property float z\n")
        myFile.write("property uchar red\n")
        myFile.write("property uchar green\n")
        myFile.write("property uchar blue\n")
        myFile.write("element face " + str(len(sphFaces) * len(points)) + "\n")
        myFile.write("property list uchar int vertex_index\n")
        myFile.write("end_header\n")

        for point, label in zip(tqdm(points), labels):
            for currSphPt in sphPts:
                currPtFlt = [aabbSize * currSphPt[0] + point[0], aabbSize * currSphPt[1] + point[1],
                             aabbSize * currSphPt[2] + point[2]]
                myFile.write(str(currPtFlt[0]) + " " + str(currPtFlt[1]) + " " + str(currPtFlt[2]) + " " + str(
                    objColors[label][0]) + " " + str(objColors[label][1]) + " " + str(objColors[label][2]) + "\n")

        offset = 0
        for i in tqdm(range(len(points))):
            for currSphFace in sphFaces:
                myFile.write("3 " + str(currSphFace[0] + offset) + " " + str(currSphFace[1] + offset) + " " + str(
                    currSphFace[2] + offset) + "\n")
            offset += len(sphPts)

    myFile.close()


# origin_path = '../CrossNet/Data/test'
origin_path = '/research/dept6/wbhu/Dataset/ScanNet/test_3d_original'
# vis_prefix = 'Exp/scannet/baseline/result/last'
# vis_prefix = 'Exp/scannet/cross_scratch_v2_0.1/result/best'
vis_prefix = 'Exp/scannet/bl_4g/result/best'
scenes = sorted(glob(join(origin_path, '*_vh_clean_2.ply')))

# gt = np.load(join(vis_prefix, 'gt.npy'))
pred = np.load(join(vis_prefix, 'benchmark.npy'))

sphPts, sphFaces = icosahedron()
for i in range(2):
    sphPts, sphFaces = subdivide(sphPts, sphFaces)

l = 0

reverseMapper = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 40])
color_palette = create_color_palette()

for s in tqdm(scenes):
    name = s.split('/')[-1][:12]
    ori = plyfile.PlyData().read(s)
    v = np.array([list(x) for x in ori.elements[0]])

    coords = np.ascontiguousarray(v[:, :3])
    # label_ori = np.ascontiguousarray(v[:, -1]).astype(np.int)
    # label_gt = gt[l:l + v.shape[0]].astype(np.int)
    # # pdb.set_trace()
    # label_gt[label_gt == 255] = 20
    # label_gt = reverseMapper[label_gt]
    label_pred = pred[l:l + v.shape[0]].astype(np.int)
    label_pred = reverseMapper[label_pred]
    l += v.shape[0]
    # ori.elements[0].data['label'] = label_pred
    ori.elements[0].data['red'] = color_palette[label_pred][:, 0]
    ori.elements[0].data['green'] = color_palette[label_pred][:, 1]
    ori.elements[0].data['blue'] = color_palette[label_pred][:, 2]
    ori.write(join(vis_prefix, name+'.ply'))
    np.savetxt(join(vis_prefix, name+'.txt'), label_pred, fmt='%d')
    # pdb.set_trace()


