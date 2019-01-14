# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.io as sio
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt


def visualization(img, dmap):
    plt.figure()

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(dmap[:, :, 0])
#    plt.colorbar()

    plt.tight_layout()
    plt.show()


def read_annotations():
    """read annotation data.

    Returns:
        count: ndarray, head count.
        position: ndarray, coordinate.
    """
    data = sio.loadmat('data\\mall_dataset\\mall_gt.mat')
    count = data['count']
    position = data['frame'][0]

    return count, position


def map_pixels(img, image_key, annotations, size):
    """map annotations to density map.

    Arguments:
        img: ndarray, img.
        image_key: int, image_key.
        annotations: ndarray, annotations.
        size: resize size.

    Returns:
        pixels: ndarray, density map.
    """
    gaussian_kernel = 15
    h, w = img.shape[:-1]
    sh, sw = size / h, size / w
    pixels = np.zeros((size, size))

    for a in annotations[image_key][0][0][0]:
        x, y = int(a[0] * sw), int(a[1] * sh)
        if y >= size or x >= size:
            print("{},{} is out of range, skipping annotation for {}".format(x, y, image_key))
        else:
            pixels[y, x] += 1

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)

    return pixels


def get_data(i, size, annotations):
    """get data accoding to the image_key.

    Arguments:
        i: int, image_key.
        size: int, input shape of network.
        annotations: ndarray, annotations.

    Returns:
        img: ndarray, img.
        density_map: ndarray, density map.
    """
    name = 'data\\mall_dataset\\frames\\seq_{}.jpg'.format(str(i + 1).zfill(6))
    img = cv2.imread(name)

    density_map = map_pixels(img, i, annotations, size // 4)

    img = cv2.resize(img, (size, size))
    img = img / 255.

    density_map = np.expand_dims(density_map, axis=-1)

    return img, density_map


def generator(indices, batch, size):
    """data generator.

    Arguments:
        indices: list, image_key.
        batch: int, batch size.
        size: int, input shape of network.

    Returns:
        images: ndarray, batch images.
        labels: ndarray, batch density maps.
    """
    count, position = read_annotations()

    i = 0
    n = len(indices)

    if batch > n:
        raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch, n))

    while True:
        if i + batch >= n:
            np.random.shuffle(indices)
            i = 0
            continue

        pool = ThreadPool(2)
        res = pool.map(lambda x: get_data(x, size, position), indices[i: i + batch])
        pool.close() 
        pool.join()

        i += batch
        images = []
        labels = []

        for r in res:
            images.append(r[0])
            labels.append(r[1])

        images = np.array(images)
        labels = np.array(labels)

        yield images, labels    


if __name__ == '__main__':
    count, position = read_annotations()
    img, density_map = get_data(10, 224, position)

    print(count[10][0])
    print(int(np.sum(density_map)))
    visualization(img, density_map)
