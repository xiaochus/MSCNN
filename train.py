# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd

from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from model import MSCNN
from data import generator


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=3,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=100,
        help="The number of train iterations.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs),int(args.size))


def train(batch, epochs, size):
    """Train the model.

    Arguments:
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        size: Integer, image size.
    """
    if not os.path.exists('model'):
        os.makedirs('model')

    model = MSCNN((size, size, 3))

    opt = RMSprop(lr=1e-3)
    model.compile(optimizer=opt, loss='mse')

    lr = ReduceLROnPlateau(monitor='loss', min_lr=1e-5)

    indices = list(range(2000))
    train, test = train_test_split(indices, test_size=0.25)

    hist = model.fit_generator(
        generator(train, batch, size),
        validation_data=generator(test, batch, size),
        steps_per_epoch=len(train) // batch,
        validation_steps=len(test) // batch,
        epochs=epochs,
        callbacks=[lr])

    model.save_weights('model\final_weights.h5')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model\\history.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv)
