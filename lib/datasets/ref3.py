from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

from tensorflow.contrib.data import Dataset


from src.base.utils import get_full_names_of_image_files

from src.base.labels_base import NexarLabelsBase


#------------------------------------------------------------------------------------------------
#   Toy example: range(20
#-------------------------------------------------------------------------------------------------
def play_range():
    dataset = tf.contrib.data.Dataset.range(20)
    dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(7)
    dataset = dataset.enumerate(start=0)
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    # create initialization ops to switch between the datasets
    # training_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:
        # get each element of the training dataset until the end is reached
        while True:
            try:
                step, data = sess.run(next_element)
                print('step: {} , data={}, len={}'.format(step,data,len(data)))
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
#------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------
# Example: serving image files from directory
#-------------------------------------------------------------------------------------------------
def parse_json_file(json_fname):
    df = NexarLabelsBase.json_to_df(json_fname)
    boxes = []
    for i, row in df.iterrows():
        if row.type == 'NexodBox':
            boxes.append(row.type_representation.as_list())

    return boxes


class PreProcessor:
    def __init__(self, target_height, target_width, rgb_means):
        self.target_height = target_height
        self.target_width = target_width
        self.rgb_means = rgb_means

    def preprocess(self, img_path, lbl_path):
        """tensorflow function"""
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        # pre-processing of image
        img_decoded.set_shape([None, None, None])
        image_resized = tf.image.resize_images(img_decoded, [self.target_height, self.target_width])
        scale_y = tf.shape(image_resized)[0] / tf.shape(img_decoded)[0]
        scale_x = tf.shape(image_resized)[1] / tf.shape(img_decoded)[1]
        image_normalized = tf.subtract(image_resized, self.rgb_means)

        # pre-processinf of bboxes
        boxes = tf.py_func(parse_json_file, [lbl_path], [tf.float64])

        return image_normalized, [scale_x,scale_y], boxes


class DatasetWraper:
    def __init__(self, images_dir, labels_dir, preprocessor):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.preprocessor = preprocessor

    def map_func(self, txt_line):
        """ Should be tensorflow function"""
        # read the img from file
        img_path = self.images_dir + txt_line
        json_path = self.labels_dir + txt_line + ".json"
        images, scales, labels = self.preprocessor.preprocess(img_path, json_path)
        return images, scales, labels


def play_image_file():
    images_file = '/test_dl/data/small_test/base/test/images.txt'
    images_dir = '/test_dl/data/small_test/raw/test/images/'
    labels_dir = '/test_dl/data/small_test/base/test/annotations/labels/'

    target_height = 272
    target_width = 480
    rgb_means = [[[123.68, 116.779, 103.939]]]
    batch_size = 7
    preprocessor = PreProcessor(target_height, target_width, rgb_means)
    dataset_wrapper = DatasetWraper(images_dir, labels_dir , preprocessor)
    print ('image files: {}'.format(images_file))

    # create TensorFlow Dataset objects
    dataset = tf.contrib.data.TextLineDataset([images_file])
    dataset = dataset.map(dataset_wrapper.map_func)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(1)
    dataset = dataset.padded_batch(batch_size,([target_height,target_width,3],[2], [None,4]))  # allow arbitrary number of boxes

    dataset = dataset.enumerate(start=0)
    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    # create initialization ops to switch between the datasets
    # training_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:
        # get each element of the training dataset until the end is reached
        while True:
            try:
                step, data = sess.run(next_element)
                print('step: {} , data={}/{}/{}'.format(step,data[0].shape,data[1].shape,data[2].shape))
                print('    {}'.format(data[2][0]))

            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
#------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
# Example: Using tfrecords ---> small POC for images and boxes
#-------------------------------------------------------------------------------------------------
def py_func_decode_json(json_string):
    json_dict = json.loads(str(json_string))

    x0 = [float(b['type_representation']['x0']) for b in json_dict]
    y0 = [float(b['type_representation']['y0']) for b in json_dict]
    x1 = [float(b['type_representation']['x1']) for b in json_dict]
    y1 = [float(b['type_representation']['y1']) for b in json_dict]

    return [x0, y0, x1, y1]


# Transforms a scalar string `example_proto` into a pair of a scalar string and
# a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=""),
      'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
      'label/json_string' : tf.FixedLenFeature((), tf.string, default_value="")}

  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.image.decode_image(parsed_features["image/encoded"])
  lbl = parsed_features["label/json_string"]
  boxes = tf.py_func(py_func_decode_json, [lbl], [tf.float64,tf.float64,tf.float64,tf.float64])

  return img, boxes

def play_tfrecords():
    batch_size = 32
    tfrecords_file = '/test_dl/data/small_test/raw/test/tfrecords_tf/images_0_of_1.tfrecord'
    filenames = [tfrecords_file]

    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(2)
    dataset = dataset.padded_batch(batch_size,([720,1280,3],[4,None]))  # allow arbitrary number of boxes
    dataset = dataset.enumerate(start=0)
    iterator = dataset.make_one_shot_iterator()


    # create TensorFlow Iterator object
    #iterator = Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()

    # create initialization ops to switch between the datasets
    # training_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:

        # initialize the iterator on the training data
        #sess.run(training_init_op)

        # get each element of the training dataset until the end is reached
        while True:
            try:
                step, (img, boxes) = sess.run(next_element)
                #lbl_json = json.loads(lbl[0])
                print('step: {}, shape:{}, boxes: {}'.format(step,img.shape,boxes.shape))
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break

def path2imagestring(img_path):
    """tensorflow function"""
    img_file = tf.read_file(img_path)
    #img_decoded = tf.image.decode_image(img_file, channels=3)

    # pre-processing of image
    #img_decoded.set_shape([None, None, None])

    return img_file

def play_dir_files():
    images_dir = '/test_dl/data/NEXAREAR_small/raw/test/images'
    image_files = get_full_names_of_image_files(images_dir)

    dataset = Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(path2imagestring)
    dataset = dataset.batch(20)

    iterator = dataset.make_one_shot_iterator()

    next_element = iterator.get_next()

    # create initialization ops to switch between the datasets
    # training_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:
        # get each element of the training dataset until the end is reached
        while True:
            try:
                data = sess.run(next_element)
                print('data={}/{}'.format(data.shape,len(data[0])))

            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break


if __name__ == "__main__":
    #play_range()

    play_dir_files()

    #play_image_file()
    #
    #play_tfrecords()
    exit(0)