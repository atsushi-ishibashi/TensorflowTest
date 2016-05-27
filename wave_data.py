# -*- coding: utf-8 -*-

import numpy
import csv

class DataSet(object):
  def __init__(self, images, labels, one_hot=False):
    assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,labels.shape))
    self._num_examples = labels.shape[0]#images.shape[0]
    print "self._num_examples:%d"%self._num_examples
    images = images.astype(numpy.float32)
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # assert images.shape[3] == 1
    #images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    # images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()
  VALIDATION_SIZE = 0
  TEST_SIZE = 100
  imageFile = open("test_image_swt_2.csv","rU")
  labelFile = open("test_label_swt_2.csv","rU")
  imageDataReader = csv.reader(imageFile)
  labelDataReader = csv.reader(labelFile)
  imageData = numpy.array(list(imageDataReader))
  labelData = numpy.array(list(labelDataReader))
  imageFile.close()
  labelFile.close()
  length = labelData.shape[0]
  validation_images = imageData[:VALIDATION_SIZE]
  validation_labels = labelData[:VALIDATION_SIZE]
  test_images = imageData[(length-TEST_SIZE):]
  test_labels = labelData[(length-TEST_SIZE):]
  train_images = imageData[VALIDATION_SIZE:(length-TEST_SIZE)]
  train_labels = labelData[VALIDATION_SIZE:(length-TEST_SIZE)]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets
