import numpy as np
import os
import pickle

data_path = "/Users/surya/tech/PycharmProjects/ANN/data/"

img_size = 32

num_channels = 3

img_size_flattened = img_size * img_size * num_channels

num_classes = 10

# Number of files for the training set
_num_files_train = 5

# Number of images per file
_images_per_file = 10000

# Number of images in the training set
_num_images_train = _num_files_train * _images_per_file

def _get_file_path(filename = ""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding = "bytes")

    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array
    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    data = _unpickle(filename)

    raw_images = data[b'data']

    # Get the class number for each image (Y)
    cls = np.array(data[b'labels'])

    # Convert the images
    images = _convert_images(raw_images)

    return images, cls

############################################
# Public Functions
############################################

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def load_class_names():
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings
    names = [x.decode('utf-8') for x in raw]

    return names

def load_training_data():

    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    begin = 0

    # For each data file
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch
        num_images = len(images_batch)

        # End-index for the current batch
        end = begin + num_images

        # Store the images into the array
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array
        cls[begin:end] = cls_batch

        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

def load_test_data():
    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
