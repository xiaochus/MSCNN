# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sklearn.metrics as metrics

from model import MSCNN
from data import visualization


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)

    print('mae:%f' % mae)
    print('mse:%f' % mse)


if __name__ == '__main__':
    name = 'data\\mall_dataset\\frames\\seq_001600.jpg'
#    name = 'data\\timg3.jpg'

    model = MSCNN((224, 224, 3))
    model.load_weights('model\\final_weights.h5')

    img = cv2.imread(name)
    img = cv2.resize(img, (224, 224))
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    dmap = model.predict(img)[0][:, :, 0]
    dmap = cv2.GaussianBlur(dmap, (15, 15), 0)

    visualization(img[0], dmap)
    print('count:', int(np.sum(dmap)))
