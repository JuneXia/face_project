import os
import sys
sys.path.append('/disk1/home/xiaj/dev/alg_verify/face/face_project')

import tensorflow as tf

from data_loader.data_generator import TFDataGenerator as DataGen
# from data_loader.data_generator import TFRecordDataGenerator as DataGen
# from models.resnet import ResnetModel as Model
from models.inception_resnet_v2 import InceptionResnetV2 as Model
# from trainers.resnet_trainer import ResnetTrainer as Trainer
from trainers.inception_resnet_trainer import InceptionResnetTrainer as Trainer
from utils.config import process_config
from utils.dirs import create_dirs
# from utils.logger import Logger
from utils.logger import Summary as Logger
from utils.utils import get_args
from utils import dataset as Datset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    sys.argv = ['src/faceid_inception_resnet.py',
                '--config', '../configs/inception_resnet_VggfaceGCWebface.json',
                ]

    try:
        args = get_args()
        print(args)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])


    # create tensorflow session
    confproto = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False,
                               graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0))
                               )
    confproto.gpu_options.allow_growth = True
    sess = tf.Session(config=confproto)

    if config.debug == True:
        end_idx = 10000
    else:
        end_idx = None
    # images_path, images_label, val_images_path, val_images_label = Datset.load_feedata(config.train_data_path, shuffle=False, end_idx=end_idx)

    # create your data generator
    # train_data = DataGen(sess, images_path, images_label, imshape=config.state_size, phase_train=True)

    imshape = (160, 160, 3)
    tfrecords = '/disk1/home/xiaj/dev/alg_verify/face/face_project/VGGFace2-GCWebFace_mainland_HongkongTaiwan_JapanKorea.tfrecords'
    train_data = Datset.TFRecordDataGenerator(sess, tfrecords=tfrecords, imshape=imshape, batch_size=config.batch_size, phase_train=True, repeat=-1)
    train_data.set_train_augment(random_crop=0, random_rotate=False, random_left_right_flip=False, standardization=0)

    # val_data = DataGen(sess, val_images_path, val_images_label, imshape=config.state_size, phase_train=False)

    # create an instance of the model you want
    # model = Model(len(set(images_label)), config)
    model = Model(12000, config)

    # create tensorboard logger
    # logger = Logger(sess, config)
    # summ_scalar, summ_hist = model.get_summary_dict()
    # logger.set_summary_tensor(summ_scalar, summ_hist)
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, config, logger, train_data, val_data=None)

    # load model if exists
    # model.load(sess)

    # here you train your model
    trainer.train(max_epochs=config.max_epochs, epoch_size=config.epoch_size)

    sess.close()

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('main end')


if __name__ == '__main__':
    main()


# 动态学习率
# evaluation
# 保存模型