import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]


import tensorflow as tf
from utils import dataset as Datset


class TFDataGenerator:
    def __init__(self, sess, images_path, images_label, imshape, batch_size=16, phase_train=True):
        """
        :param sess:
        :param images_path:
        :param images_label:
        :param imshape:
        :param batch_size:
        :param phase_train:
        """
        # super(TFDataGenerator, self).__init__()
        self.sess = sess
        # if config.debug == 1:
        #     end_idx = 10000
        # else:
        #     end_idx = None
        # image_list, label_list, val_image_list, val_label_list = Datset.load_feedata(config.train_data_path, end_idx=end_idx)

        imparse = Datset.ImageParse(imshape=imshape)
        if phase_train:
            parse_func = imparse.train_parse_func
        else:
            parse_func = imparse.validation_parse_func

        filenames = tf.constant(images_path)
        filelabels = tf.constant(images_label, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        dataset = dataset.map(parse_func, num_parallel_calls=4)  # tf.data.experimental.AUTOTUNE
        dataset = dataset.shuffle(buffer_size=20000,
                                  # seed=tf.compat.v1.set_random_seed(666),
                                  reshuffle_each_iteration=True
                                  ).batch(batch_size).prefetch(buffer_size=5000)  # repeat 不指定参数表示允许无穷迭代

        self.iterator = dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        self.sess.run(self.iterator.initializer)

    def next_batch(self):
        try:
            images, labels = self.sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.iterator.initializer)
            images, labels = self.sess.run(self.next_element)

        return images, labels


class TFRecordDataGenerator:
    def __init__(self, sess, config, tfrecords, phase_train=True):
        """
        TODO: 可以考虑将数据增强部分参数放入初始化参数中。
        :param sess:
        :param config:
        :param images_path:
        :param images_label:
        :param phase_train:
        """
        self.sess = sess
        imparse = Datset.ImageParse(imshape=config.state_size)
        if phase_train:
            parse_func = imparse.train_parse_func
        else:
            parse_func = imparse.validation_parse_func

        filenames = tf.train.match_filenames_once(tfrecords)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parse_func, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=2000,
                                  # seed=tf.compat.v1.set_random_seed(666),
                                  reshuffle_each_iteration=True
                                  ).batch(config.batch_size)  # repeat 不指定参数表示允许无穷迭代

        # # super(TFDataGenerator, self).__init__()
        # self.sess = sess
        # # if config.debug == 1:
        # #     end_idx = 10000
        # # else:
        # #     end_idx = None
        # # image_list, label_list, val_image_list, val_label_list = Datset.load_feedata(config.train_data_path, end_idx=end_idx)
        #
        # imparse = Datset.ImageParse(imshape=config.state_size)
        # if phase_train:
        #     parse_func = imparse.train_parse_func
        # else:
        #     parse_func = imparse.validation_parse_func
        #
        # filenames = tf.constant(images_path)
        # filelabels = tf.constant(images_label, dtype=tf.int64)
        # dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        # dataset = dataset.map(parse_func, num_parallel_calls=4)  # tf.data.experimental.AUTOTUNE
        # dataset = dataset.shuffle(buffer_size=2000,
        #                           # seed=tf.compat.v1.set_random_seed(666),
        #                           reshuffle_each_iteration=True
        #                           ).batch(config.batch_size)  # repeat 不指定参数表示允许无穷迭代

        self.iterator = dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        self.sess.run(self.iterator.initializer)

    def next_batch(self):
        try:
            images, labels = self.sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.iterator.initializer)
            images, labels = self.sess.run(self.next_element)

        return images, labels


