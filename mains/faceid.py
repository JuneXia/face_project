import os
import sys

import tensorflow as tf

from data_loader.data_generator import TFDataGenerator as DataGen
from models.resnet import ResnetModel as Model
from trainers.resnet_trainer import ResnetTrainer as Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Summary as Logger
from utils.utils import get_args
from utils import dataset as Datset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    sys.argv = ['src/faceid.py',
                '--config', '../configs/resnet_VggfaceGCWebface.json',
                ]

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    confproto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    confproto.gpu_options.allow_growth = True
    sess = tf.Session(config=confproto)

    if config.debug == 1:
        end_idx = 10000
    else:
        end_idx = None
    images_path, images_label, val_images_path, val_images_label = Datset.load_feedata(config.train_data_path, end_idx=end_idx)

    # create your data generator
    train_data = DataGen(sess, config, images_path, images_label, phase_train=True)
    config.num_classes = len(set(images_label))
    val_data = DataGen(sess, config, val_images_path, val_images_label, phase_train=False)

    # create an instance of the model you want
    model = Model(config)

    # create tensorboard logger
    logger = Logger(sess, config)
    summ_scalar, summ_hist = model.get_summary_dict()
    logger.set_summary_tensor(summ_scalar, summ_hist)

    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, config, logger, train_data, val_data=val_data)

    # load model if exists
    # model.load(sess)

    # here you train your model
    trainer.train(max_epochs=config.max_epochs, epoch_size=config.epoch_size)

    sess.close()

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('main end')


if __name__ == '__main__':
    main()
