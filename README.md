# Keras on Cloud ML Engine: MNIST Multi-Layer Perceptron

## Keras MNIST MLP

Keras has a wide range of neural network/deep learning examples on [github]. Let's adapt their [MNIST example] which creates a Multi-Layer Perceptron (MLP) model to run on Google's [Cloud ML Engine].

### (Optional) Understanding the MNIST MLP example

The [MNIST dataset] is probably the most common dataset used in introductory machine learning examples these days. Basically, MNIST is a very high quality dataset that contains thousands of hand-drawn digits like:

![MNIST sample][MNIST sample]

classified by their integer labels "3", "7", etc. The idea is to train a deep (i.e. multi-layered) neural network classifies hand-drawn digits according to their numeric label. The usual implementation of the network outputs the probability that a given digit is in each of the possible digit classes. For example, it's possible the digit is poorly drawn and has a 50% chance to be a "1" and a 50% chance to be a "7".

```python
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
```
The `keras.dataset` import pulls from `https://s3.amazonaws.com/img-datasets/mnist.npz`, which is an uncompressed file containing `numpy` array(s).

```python
batch_size = 128
num_classes = 10
epochs = 20
```

* `batch_size` is the number of training images to pass through before the neural network's parameters are updated. Decrease this number (i.e. increase the number of parameter updates) to allow faster training and reduced memory usage. Increase this number to improve the estimate of the gradient.
* `num_classes` is the number of output classes (the possible digits are 0-9, so there are 10 different classes an image can belong to).
* `epochs` is the number of iterations over the entire training dataset. The accuracy, loss, etc. are calculated each time.

```python
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
The data in `mnist.npz` is already formatted (separated into training and testing/validation data) to make this line work.

```python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```
Each 28x28 image is 784 total pixels. There are 60,000 in the training set, and 10,000 in the testing (validation) set. The pixels have the values 0-255 to represent the 256 possible colors. We convert the color values to be `float32`s between 0 and 1.

```python
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
The original data contains the obvious classes for each hand-drawn digit, e.g. for a hand-drawn "6", the image is classified as the number 6. `to_categorical` changes this classification to a vector with a 1 in the 6th entry and 0s otherwise. In this so-called *one-hot encoding*, an output (prediction) vector of the form `[0,0.50,0,0,0,0,0,0.50,0,0]`, for example, means that the hand-drawn digit has a 50% chance to be a 1 and a 50% chance to be a 7.

```python
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()
```
This creates a sequentially layered neural network (meaning non-convolutional or not passed through a filter).

Dense layers are also called *fully-connected layers*, meaning all the neurons in the previous layer are connected to the following layer. The first argument is the number of neurons in the following layer, and only the first Dense layer needs to have the `input_shape` specified (the previous layer shape is inferred by keras for subsequent layers).

Dropout is technique to forget some of the parameters in the network. It sounds counterintuitive to throw away data, but the reason for this is to avoid overfitting (you can't overfit if you don't have all of the input data!).

```python
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
```
A loss function helps you measure error or its best friend accuracy, but the loss number is not itself absolute. i.e. it doesn't matter *what* the loss value is, just that it's *relatively small*. The thing we really care about is *accuracy*.

*Note*: There are two different accuracies calculated, one for the training set and one for the test (validation) set. High training accuracy means your model is really good at predicting the particular data you gave it. High validation accuracy means your model is really good at predicting data it hasn't seen yet. It's thus very important for your validation data to *accurately represent real-world data*.

```python
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
```
Trains the model. Since there are 60,000 images, the parameters are updated `60000/batch_size` times per `epoch`. Since `verbose=1`, keras outputs the loss, accuracy, validation loss, and validation accuracy for each epoch.

```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

At the end of the epochs, the model is validated against the test data, and the final validation loss and accuracy are printed.

Here is the example output:
```shell
Using TensorFlow backend.
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 6s - loss: 0.2504 - acc: 0.9226 - val_loss: 0.1202 - val_acc: 0.9606
Epoch 2/20
60000/60000 [==============================] - 6s - loss: 0.1045 - acc: 0.9675 - val_loss: 0.0799 - val_acc: 0.9754
Epoch 3/20
60000/60000 [==============================] - 6s - loss: 0.0775 - acc: 0.9766 - val_loss: 0.0746 - val_acc: 0.9745
Epoch 4/20
60000/60000 [==============================] - 6s - loss: 0.0592 - acc: 0.9820 - val_loss: 0.0872 - val_acc: 0.9771
Epoch 5/20
60000/60000 [==============================] - 6s - loss: 0.0527 - acc: 0.9846 - val_loss: 0.0737 - val_acc: 0.9808
Epoch 6/20
60000/60000 [==============================] - 6s - loss: 0.0438 - acc: 0.9866 - val_loss: 0.0810 - val_acc: 0.9783
Epoch 7/20
60000/60000 [==============================] - 6s - loss: 0.0392 - acc: 0.9884 - val_loss: 0.0778 - val_acc: 0.9821
Epoch 8/20
60000/60000 [==============================] - 6s - loss: 0.0337 - acc: 0.9900 - val_loss: 0.0734 - val_acc: 0.9839
Epoch 9/20
60000/60000 [==============================] - 6s - loss: 0.0342 - acc: 0.9901 - val_loss: 0.0820 - val_acc: 0.9828
Epoch 10/20
60000/60000 [==============================] - 6s - loss: 0.0290 - acc: 0.9916 - val_loss: 0.0839 - val_acc: 0.9826
Epoch 11/20
60000/60000 [==============================] - 6s - loss: 0.0261 - acc: 0.9926 - val_loss: 0.0983 - val_acc: 0.9809
Epoch 12/20
60000/60000 [==============================] - 6s - loss: 0.0274 - acc: 0.9928 - val_loss: 0.0968 - val_acc: 0.9837
Epoch 13/20
60000/60000 [==============================] - 6s - loss: 0.0229 - acc: 0.9935 - val_loss: 0.1056 - val_acc: 0.9814
Epoch 14/20
60000/60000 [==============================] - 6s - loss: 0.0248 - acc: 0.9934 - val_loss: 0.0990 - val_acc: 0.9816
Epoch 15/20
60000/60000 [==============================] - 6s - loss: 0.0204 - acc: 0.9942 - val_loss: 0.1034 - val_acc: 0.9832
Epoch 16/20
60000/60000 [==============================] - 6s - loss: 0.0205 - acc: 0.9945 - val_loss: 0.1028 - val_acc: 0.9844
Epoch 17/20
60000/60000 [==============================] - 6s - loss: 0.0203 - acc: 0.9946 - val_loss: 0.1127 - val_acc: 0.9813
Epoch 18/20
60000/60000 [==============================] - 6s - loss: 0.0194 - acc: 0.9948 - val_loss: 0.1001 - val_acc: 0.9848
Epoch 19/20
60000/60000 [==============================] - 6s - loss: 0.0198 - acc: 0.9952 - val_loss: 0.1277 - val_acc: 0.9816
Epoch 20/20
60000/60000 [==============================] - 6s - loss: 0.0197 - acc: 0.9950 - val_loss: 0.1106 - val_acc: 0.9836
Test loss: 0.110611914037
Test accuracy: 0.9836
```

## How to satisfy Cloud ML Engine project structure requirements

Once we're done, the basic project structure will look something like this:
```shell
.
├── README.md
├── data
│   └── mnist.pkl
├── setup.py
└── trainer
    ├── __init__.py
    └── mnist_mlp.py
```

### (Prerequisite) Install Google Cloud SDK

The best way to get started using Cloud ML Engine is to use the tools provided in the [Google Cloud SDK].

Install the SDK, then run:
```shell
gcloud init
```
and then set up your credentials quickly via web browser:
```shell
gcloud auth application-default login
```

Now that the Cloud SDK is set up, you can check your Cloud ML Engine available
models:
```shell
gcloud ml-engine models list
```
You should see `Listed 0 items.` because we haven't created any ML Engine models
yet.

### Download the data once and for all

The code from the keras github [MNIST example] downloads the MNIST data *every time* it is run. That's impractical/expensive for large datasets, so we will get a pickled version of the MNIST data to illustrate a [more general data preparation process] you might follow in your own projects.

```shell
mkdir data
curl -O https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
gzip -d mnist.pkl.gz
mv mnist.pkl data/
```

### Upload the data to a Google Cloud Storage bucket

Cloud ML Engine works by using resources available in the cloud, so the training data needs to be placed in such a resource. For this example, we'll use [Google Cloud Storage], but it's possible to use other resources like [BigQuery]. Make a bucket (names must be globally unique) and place the data in there:

```shell
gsutil mb gs://your-bucket-name
gsutil cp -r data/mnist.pkl gs://your-bucket-name/data/mnist.pkl
```

### Project configuration file: `setup.py`

The `setup.py` file is run on the Cloud ML Engine server to install packages/dependencies and set a few options.

```python
'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='mnist_mlp',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='MNIST MLP keras model on Cloud ML Engine',
      author='Your Name',
      author_email='you@example.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)
```
Technically, Cloud ML Engine [requires a TensorFlow application to be pre-packaged] so that it can install it on the servers it spins up. However, if you supply a `setup.py` in the project root directory, then Cloud ML Engine will actually create the package for you.

### Create the `__init__.py` file

For the Cloud ML Engine to create a package for your module, it's absolutely for your project to contain `trainer/__init__.py`, but it can be empty.
```shell
mkdir trainer
touch trainer/__init__.py
```
Without `__init__.py` local training will work, but when you try to submit a job to Cloud ML Engine, you will get the cryptic error message:
```shell
ERROR: (gcloud.ml-engine.jobs.submit.training) [../trainer] is not a valid Python package because it does not contain an `__init__.py` file. Please create one and try again.
```

### Fix `mnist_mlp.py` to reflect its new clouded reality

```python
'''A modification of the mnist_mlp.py example on the keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.
'''

from __future__ import print_function

import argparse
import pickle # for handling the new data source
import h5py # for saving the model
import keras
from datetime import datetime # for filename conventions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.python.lib.io import file_io # for better file I/O
import sys

batch_size = 128
num_classes = 10
epochs = 20

# Create a function to allow for different training data and other options
def train_model(train_file='data/mnist.pkl',
                job_dir='./tmp/mnist_mlp', **args):
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
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
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
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
```

### Run the model with python (locally)
```shell
python trainer/mnist_mlp.py --job-dir ./tmp/mnist_mlp --train-file data/mnist.pkl
```

### Run the model with gcloud
First, export some environment variables:
```shell
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_mlp_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1
```

For more details on the following commands, see the [`gcloud ml-engine` documentation].

To run the model locally:
```shell
gcloud ml-engine local train \
  --job-dir $JOB_DIR \
  --module-name trainer.mnist_mlp \
  --package-path ./trainer \
  -- \
  --train-file ./data/mnist.pkl
```
*Note*: The order of the options is important. In particular, the extra `--` is required to signal that the options following it should be passed to the module directly at run-time.

To submit a job to Cloud ML Engine:
```shell
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.0 \
    --module-name trainer.mnist_mlp \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-file gs://$BUCKET_NAME/data/mnist.pkl
```
You can check the [job status], where logs are also available.

### (Optional) Hyperparameter tuning

A **hyperparameter** can be thought of as a parameter for a model that is set *before* the model is trained -- contrast with **weights** and **biases**, which are set *during* the training process.

Cloud ML Engine can do [hyperparameter tuning], i.e. running training multiple times to try to figure out good values for hyperparameters. To make this work, the trainer module has to take in the hyperparameters as arguments.

#### Example: tuning the Dropout layers

The file [mnist_mlp_hpt.py](mnist/trainer/mnist_mlp_hpt.py) contains the modified code to accept the `dropout-one` and `dropout-two` hyperparameter arguments.

Additionally, we need a file `hptuning_config.yaml` that explains what `dropout-one` and `dropout-two` are to the tuner. Basically, these are doubles between `0.1` and `0.5`, which correspond to dropping out 10% to 50% of the incoming parameters from the previous layer. The doubles are chosen to maximize the `accuracy` metric. `UNIT_REVERSE_LOG_SCALE` is chosen so that it checks values more densely on the bottom end of the range, since the original values were `0.2`. Four trials are run, with a maximum of two running at any given time:

```yaml
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 4
    maxParallelTrials: 2
    params:
      - parameterName: dropout-one
        type: DOUBLE
        minValue: 0.1
        maxValue: 0.5
        scaleType: UNIT_REVERSE_LOG_SCALE
      - parameterName: dropout-two
        type: DOUBLE
        minValue: 0.1
        maxValue: 0.5
        scaleType: UNIT_REVERSE_LOG_SCALE
```

Some additional options need to be passed to `gcloud`, namely `config` (specifying the hyperparameter config file) and the new hyperparameter arguments `dropout-one` and `dropout-two`:
```shell
export BUCKET_NAME=your-bucket-name
export JOB_NAME="mnist_mlp_hpt_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1
export HPTUNING_CONFIG=hptuning_config.yaml
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.0 \
    --config $HPTUNING_CONFIG \
    --module-name trainer.mnist_mlp_hpt \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-file gs://$BUCKET_NAME/data/mnist.pkl \
    --dropout-one 0.2 \
    --dropout-two 0.2
```
To see the values, check the [job status] which contains the logs and the hyperparameters that Cloud ML Engine found.

#### Acknowledgements

I would like to thank Fuyang Liu for a [tutorial that helped greatly] in understanding how to use keras with Cloud ML Engine.


[BigQuery]: https://cloud.google.com/bigquery/
[Cloud ML Engine]: https://cloud.google.com/ml-engine/
[`gcloud ml-engine` documentation]: https://cloud.google.com/sdk/gcloud/reference/ml-engine/
[github]: https://github.com/fchollet/keras/tree/master/examples
[Google Cloud SDK]: https://cloud.google.com/sdk/
[Google Cloud Storage]: https://cloud.google.com/storage/
[hyperparameter tuning]: https://cloud.google.com/ml-engine/docs/concepts/hyperparameter-tuning-overview
[job status]: https://console.cloud.google.com/mlengine/jobs/
[MNIST dataset]: http://yann.lecun.com/exdb/mnist/
[MNIST example]: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
[MNIST sample]: http://myselph.de/mnistExamples.png
[more general data preparation process]: https://cloud.google.com/ml-engine/docs/concepts/data-prep
[requires a TensorFlow application to be pre-packaged]: https://cloud.google.com/ml-engine/docs/how-tos/packaging-trainer
[tutorial that helped greatly]: http://liufuyang.github.io/2017/04/02/just-another-tensorflow-beginner-guide-4.html
