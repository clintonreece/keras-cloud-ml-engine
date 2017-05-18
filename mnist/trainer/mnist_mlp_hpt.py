"""A modification of the mnist_mlp.py example on the Keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.

This file also incorporates hyperparameter tuning on the Dropout layers.
"""

from __future__ import print_function

import argparse
import pickle  # for handling the new data source
import h5py  # for saving the model
import keras
from datetime import datetime  # for filename conventions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.python.lib.io import file_io  # for better file I/O
import sys

batch_size = 128
num_classes = 10
epochs = 20


# Create a function to allow for different training data and other options
def train_model(train_file='data/mnist.pkl',
                job_dir='./tmp/mnist_mlp',
                dropout_one=0.2,
                dropout_two=0.2,
                **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    # Reading in the pickle file. Pickle works differently with Python 2 vs 3
    f = file_io.FileIO(train_file, mode='r')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = data

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout_one))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_two))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save the model locally
    model.save('model.h5')

    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help='Cloud storage bucket to export the model and store temp files')
    parser.add_argument(
        '--dropout-one',
        help='Dropout hyperparameter after the first dense layer')
    parser.add_argument(
        '--dropout-two',
        help='Dropout hyperparameter after the second dense layer')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
