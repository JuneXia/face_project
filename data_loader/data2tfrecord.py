import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('/disk1/home/xiaj/dev/alg_verify/face/face_project')

from utils import dataset as Datset
from utils import tools
import cv2
from data_loader import data_generator as DatGen


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


def write_tfrecord():
    pass


# 将mnist数据写入tfrecord文件
if __name__ == '__main__1':
    data_path = "/disk1/home/xiaj/res/face/training/VGGFace2_disk2-GCWebFace_mainland_HongkongTaiwan_JapanKorea.csv"
    images_path, images_label, val_images_path, val_images_label = Datset.load_feedata(data_path, shuffle=False, end_idx=300000)

    # data_path = '/home/xiajun/dataset/tmp'
    # data_path = '/home/xiajun/dataset/gc_together/origin_gen90_align160_margin32'
    # data_path = '/home/xiajun/res/face/CelebA/Experiment/img_align_celeba_identity'
    # images_path, images_label = Datset.load_dataset(data_path, shuffle=False, validation_ratio=0.0)

    confproto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    confproto.gpu_options.allow_growth = True
    sess = tf.Session(config=confproto)

    imshape = (182, 182, 3)
    # train_data = DatGen.TFDataGenerator(sess, images_path, images_label, imshape=(160, 160, 3), batch_size=1, phase_train=True)
    train_data = Datset.TFDataGenerator(sess, images_path, images_label, imshape=imshape, batch_size=1, phase_train=True, repeat=1)
    train_data.set_train_augment(random_crop=0, random_rotate=False, random_left_right_flip=False, standardization=0)

    debug_show = False
    with tf.python_io.TFRecordWriter("VGGFace2-GCWebFace_mainland_HongkongTaiwan_JapanKorea.tfrecords") as writer:
        count = 0
        while True:
            try:
                image, label = train_data.next_batch()
                print(image.shape, label.shape, label)
            except Exception as err:
                print(err)
                break

            count += 1

            # image, label = train_data.next_batch()
            example = _make_example(imshape[0], label[0], image[0])
            writer.write(example.SerializeToString())
            tools.view_bar('TFRecord Writing: ', count, len(images_path))

            if debug_show:
                img = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
                cv2.imshow('show', img)
                cv2.waitKey(0)
        print('')


# 解析一个TFRecord的方法。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [160, 160, 3])
    labels = tf.cast(features['label'],tf.int32)
    #pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels


if __name__ == '__main__2':
    train_files = tf.train.match_filenames_once("output.tfrecords")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    image_size = 299  # 定义神经网络输入层图片的大小。
    batch_size = 1  # 定义组合数据batch的大小。
    shuffle_buffer = 2  # 定义随机打乱数据时buffer的大小。

    # 定义读取训练数据的数据集。
    dataset = tf.data.TFRecordDataset(train_files)
    dataset = dataset.map(parser)

    # 对数据进行shuffle和batching操作。这里省略了对图像做随机调整的预处理步骤。
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

    # 重复NUM_EPOCHS个epoch。
    NUM_EPOCHS = 10
    dataset = dataset.repeat(NUM_EPOCHS)

    # 定义数据集迭代器。
    iterator = dataset.make_initializable_iterator()
    # image_batch, label_batch = iterator.get_next()
    next_element = iterator.get_next()

    confproto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    confproto.gpu_options.allow_growth = True
    sess = tf.Session(config=confproto)

    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))

    sess.run(iterator.initializer)

    count = 0
    while True:
        try:
            image, label = sess.run(next_element)
        except Exception as err:
            print(err)
            break

        count += 1


        if True:
            img = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
            print(img.shape)
            img = img.astype(np.uint8)
            cv2.imshow('show', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    # tfrecords = '/home/xiajun/dev/alg_verify/face/face_project/data_loader/gc_together_origin_gen90_align160_margin32.tfrecords'
    tfrecords = '/disk1/home/xiaj/dev/alg_verify/face/face_project/VGGFace2-GCWebFace_mainland_HongkongTaiwan_JapanKorea.tfrecords'
    confproto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    confproto.gpu_options.allow_growth = True
    sess = tf.Session(config=confproto)

    g1 = tf.Graph()
    with g1.as_default():
        v1 = tf.get_variable('test1', shape=[1])
        test_constant1 = tf.constant([1, 2, 3, 4, 5], name='const1')
    v2 = tf.get_variable('test2', shape=[1])
    test_constant2 = tf.constant([1, 2, 3, 4, 5], name='const2')


    imshape = (160, 160, 3)
    train_data = Datset.TFRecordDataGenerator(sess, tfrecords=tfrecords, imshape=imshape, batch_size=50, phase_train=True, repeat=-1)
    train_data.set_train_augment(random_crop=0, random_rotate=False, random_left_right_flip=False, standardization=0)

    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))

    count = 0
    debug_show = True
    while True:
        try:
            image, label = train_data.next_batch()
            print(image.shape, label.shape, label)
        except Exception as err:
            print(err)
            break

        count += 1

        if debug_show:
            img = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
            print(img.shape)
            img = img * 128 + 127.5
            img = img.astype(np.uint8)
            cv2.imshow('show', img)
            cv2.waitKey(0)
    print('')
