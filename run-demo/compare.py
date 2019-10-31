from __future__ import division

import argparse
from pathlib import Path

import cv2
import numpy as np
import tabulate

import face_model


def compare(model, folder):
    p = Path(folder)
    imgs_path = list(p.rglob('*'))
    imgs_name = [x.name for x in imgs_path]
    imgs = [cv2.imread(str(x)) for x in imgs_path]

    table_header = ['name'] + imgs_name

    faces = [model.get_input(x) for x in imgs]
    embeddings = [model.get_feature(x) for x in faces]

    data = []

    for i, img1 in enumerate(imgs_path):
        data.append([img1.name])
        dist = np.sum(np.square(embeddings[i] - embeddings), axis=1).tolist()
        data[i].extend(dist)

    print(tabulate.tabulate(data, table_header))


def get_args():
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    # parser.add_argument('--img1')
    # parser.add_argument('--img2')
    parser.add_argument('--img_folder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model = face_model.FaceModel(args)

    dist = compare(model, args.img_folder)
    print(dist)
