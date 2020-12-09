'''
Codes were heavily borrowed from:
https://github.com/zeusees/HyperFAS
'''

import math
import cv2
import time
import numpy as np
import os
import glob
import pandas as pd
import keras
from keras.layers import *
import tensorflow as tf

from mtcnn import MTCNN

from keras.models import load_model

model = load_model("/model/fas.h5")

def process(ans):
    ans['score'] = ans['score'].str.replace('[', '')
    ans['score'] = ans['score'].str.replace(']', '')
    ans['score'] = ans['score'].astype('float64')
    for i in ans.iterrows():
        name = i[1]
        bool = ans['name'].str.contains(name['name'][:-6])
        data = ans[bool]
        if data.empty:
            continue
        else:
            ans = ans[~ bool]
            data = data.reset_index(drop=True)
            out_put_data = data.iloc[data['score'].idxmax()]
            mark = True
            for j in data.iterrows():
                if j[1]['name'] == out_put_data['name'] and mark:
                    n = out_put_data['name']
                    print('move: ' + out_put_data['name'])
                    os.rename('runs/' + n, 'res/' + n[:-6] + '.png')
                    mark = False
                else:
                    print('delete: ' + j[1]['name'])
                    os.remove('runs/' + j[1]['name'])

def load_mtcnn_model(model_path):
    mtcnn = MTCNN(model_path)
    return mtcnn


def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X, (224, 224)) - 127.5) / 127.5
    t = model.predict(np.array([X]))[0]
    time_end = time.time()
    return t


def test_camera(mtcnn, index=0):
    PATH = "/runs/"
    paths = glob.glob(os.path.join(PATH, '*.png'))
    paths.sort()
    ans = []
    for path in paths:
        frame = cv2.imread(path)

        image = frame

        if image is not None:

            img_size = np.asarray(image.shape)[0:2]

            bounding_boxes, scores, landmarks = mtcnn.detect(image)

            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                for det, pts in zip(bounding_boxes, landmarks):
                    det = det.astype('int32')
                    det = np.squeeze(det)
                    y1 = int(np.maximum(det[0], 0))
                    x1 = int(np.maximum(det[1], 0))
                    y2 = int(np.minimum(det[2], img_size[1] - 1))
                    x2 = int(np.minimum(det[3], img_size[0] - 1))

                    w = x2 - x1
                    h = y2 - y1
                    _r = int(max(w, h) * 0.6)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    x1 = cx - _r
                    y1 = cy - _r

                    x1 = int(max(x1, 0))
                    y1 = int(max(y1, 0))

                    x2 = cx + _r
                    y2 = cy + _r

                    h, w, c = frame.shape
                    x2 = int(min(x2, w - 2))
                    y2 = int(min(y2, h - 2))

                    _frame = frame[y1:y2, x1:x2]
                    score = test_one(_frame)
                    ans.append([os.path.basename(path), score])
                    print(score)

    df = pd.DataFrame(columns=["name", "score"], data=ans)
    df.to_csv("/res/ans.csv")
    return df


if __name__ == '__main__':
    mtcnn = load_mtcnn_model("/model/mtcnn.pb")
    ans = test_camera(mtcnn)
    process(ans)
